import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import build_model, get_eval_transform, GradCAM, get_base_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "models" / "best_model.pt"
META_PATH = BASE_DIR / "output" / "models" / "meta.json"

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

CLASS_NAMES = meta["class_names"]
IMG_SIZE = meta["img_size"]
NUM_CLASSES = meta["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES, device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

eval_transform = get_eval_transform(IMG_SIZE)

# GradCAM setup
raw_model = get_base_model(model)
gradcam = GradCAM(raw_model, raw_model.features[-1])

# History
prediction_history = []

BAR_COLORS = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffd700",
              "#ff7b72", "#79c0ff", "#7ee787", "#e3b341", "#ffa657"]

HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Teste Visual CNN</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d1117; color: #c9d1d9;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
        h1 { text-align: center; font-size: 1.8em; margin-bottom: 8px; color: #e6edf3; }
        .subtitle { text-align: center; color: #8b949e; margin-bottom: 30px; font-size: 0.95em; }

        /* Drop zone */
        .drop-zone {
            border: 2px dashed #30363d; border-radius: 12px;
            padding: 50px 20px; text-align: center;
            cursor: pointer; transition: all 0.2s;
            background: #161b22; margin-bottom: 20px;
        }
        .drop-zone:hover, .drop-zone.drag-over {
            border-color: #58a6ff; background: #1a2332;
        }
        .drop-zone p { color: #8b949e; font-size: 1.1em; }
        .drop-zone .icon { font-size: 2.5em; margin-bottom: 10px; display: block; }
        .drop-zone input { display: none; }

        .btn {
            display: block; margin: 0 auto 30px; padding: 12px 40px;
            border: none; border-radius: 8px; background: #238636;
            color: white; font-size: 1em; cursor: pointer; transition: background 0.2s;
        }
        .btn:hover { background: #2ea043; }
        .btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

        /* Results */
        .results { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
        .img-box {
            flex: 1; min-width: 280px; background: #161b22;
            border-radius: 10px; overflow: hidden; border: 1px solid #30363d;
        }
        .img-box img { width: 100%; display: block; }
        .img-box .label {
            padding: 8px 12px; font-size: 0.85em; color: #8b949e;
            text-align: center; border-top: 1px solid #30363d;
        }

        /* Predictions */
        .predictions { margin-bottom: 30px; }
        .predictions h3 { margin-bottom: 15px; color: #e6edf3; }
        .pred-row {
            display: flex; align-items: center; margin-bottom: 10px;
            background: #161b22; border-radius: 8px; padding: 10px 15px;
            border: 1px solid #30363d;
        }
        .pred-name { width: 140px; font-weight: 600; font-size: 0.95em; }
        .pred-bar-bg {
            flex: 1; height: 24px; background: #21262d;
            border-radius: 6px; overflow: hidden; margin: 0 15px;
        }
        .pred-bar {
            height: 100%; border-radius: 6px;
            transition: width 0.6s ease-out;
        }
        .pred-conf { width: 60px; text-align: right; font-weight: 600; font-size: 0.95em; }

        /* History */
        .history { border-top: 1px solid #21262d; padding-top: 20px; }
        .history h3 { margin-bottom: 15px; color: #e6edf3; }
        .hist-grid { display: flex; flex-wrap: wrap; gap: 12px; }
        .hist-item {
            width: 120px; background: #161b22; border-radius: 8px;
            overflow: hidden; border: 1px solid #30363d; text-align: center;
        }
        .hist-item img { width: 100%; height: 100px; object-fit: cover; }
        .hist-item .info { padding: 6px; font-size: 0.75em; }
        .hist-item .info .cls { color: #58a6ff; font-weight: 600; }
        .hist-item .info .conf { color: #8b949e; }

        .loading { text-align: center; color: #8b949e; font-size: 1.1em; display: none; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Teste Visual da CNN</h1>
        <p class="subtitle">EfficientNet V2-S + GeM Pooling | {{ num_classes }} classes | {{ img_size }}x{{ img_size }}</p>

        <form id="upload-form" action="/" method="post" enctype="multipart/form-data">
            <div class="drop-zone" id="drop-zone">
                <span class="icon">&#128444;</span>
                <p>Arraste uma imagem aqui ou clique para selecionar</p>
                <input type="file" name="file" id="file-input" accept="image/*" required>
            </div>
            <button type="submit" class="btn" id="submit-btn" disabled>Analisar Imagem</button>
        </form>

        <div class="loading" id="loading">Analisando imagem...</div>

        {% if result %}
        <div class="results">
            <div class="img-box">
                <img src="/uploads/{{ result.filename }}">
                <div class="label">Original</div>
            </div>
            <div class="img-box">
                <img src="/uploads/gradcam_{{ result.filename }}">
                <div class="label">GradCAM — onde o modelo olha</div>
            </div>
        </div>

        <div class="predictions">
            <h3>Top-{{ result.predictions|length }} Predicoes</h3>
            {% for pred in result.predictions %}
            <div class="pred-row">
                <span class="pred-name" style="color: {{ pred.color }}">{{ pred.name }}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: {{ pred.pct }}%; background: {{ pred.color }};"></div>
                </div>
                <span class="pred-conf" style="color: {{ pred.color }}">{{ pred.pct }}%</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if history %}
        <div class="history">
            <h3>Historico ({{ history|length }})</h3>
            <div class="hist-grid">
                {% for h in history|reverse %}
                <div class="hist-item">
                    <img src="/uploads/{{ h.filename }}">
                    <div class="info">
                        <div class="cls">{{ h.class_name }}</div>
                        <div class="conf">{{ h.confidence }}%</div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const submitBtn = document.getElementById('submit-btn');
        const form = document.getElementById('upload-form');
        const loading = document.getElementById('loading');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            fileInput.files = e.dataTransfer.files;
            if (fileInput.files.length > 0) {
                submitBtn.disabled = false;
                dropZone.querySelector('p').textContent = fileInput.files[0].name;
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                submitBtn.disabled = false;
                dropZone.querySelector('p').textContent = fileInput.files[0].name;
            }
        });
        form.addEventListener('submit', () => {
            loading.style.display = 'block';
            submitBtn.disabled = true;
        });
    </script>
</body>
</html>
"""

app = Flask(__name__, static_folder="uploads")
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


def generate_gradcam_overlay(image_path, tensor, filename):
    inp = tensor.unsqueeze(0).to(device)
    if device.type == "cuda":
        inp = inp.to(memory_format=torch.channels_last)

    heatmap, pred_cls, probs = gradcam(inp)

    # Load original image for overlay
    orig = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    orig_arr = np.array(orig).astype(np.float32) / 255.0

    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = np.clip(0.55 * orig_arr + 0.45 * heatmap_colored, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_img.save(UPLOAD_FOLDER / f"gradcam_{filename}")

    return probs[0].cpu().numpy()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = file.filename
            image_path = UPLOAD_FOLDER / filename
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            tensor = eval_transform(img)
            probs = generate_gradcam_overlay(image_path, tensor, filename)

            # Top-K predictions
            top_k = min(len(CLASS_NAMES), 5)
            top_idx = np.argsort(probs)[::-1][:top_k]
            predictions = []
            for rank, idx in enumerate(top_idx):
                pct = round(float(probs[idx]) * 100, 1)
                color = BAR_COLORS[idx % len(BAR_COLORS)]
                predictions.append({"name": CLASS_NAMES[idx], "pct": pct, "color": color})

            result = {"filename": filename, "predictions": predictions}

            # Add to history
            prediction_history.append({
                "filename": filename,
                "class_name": CLASS_NAMES[top_idx[0]],
                "confidence": round(float(probs[top_idx[0]]) * 100, 1),
            })

    return render_template_string(
        HTML_PAGE,
        result=result,
        history=prediction_history[-20:],
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_static_file(filename)


if __name__ == "__main__":
    print(f"Model loaded on {device} | {NUM_CLASSES} classes | {IMG_SIZE}x{IMG_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
    app.run(host="0.0.0.0", port=5000, debug=False)
