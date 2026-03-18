import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, render_template_string, request
from PIL import Image

from train import build_model, get_eval_transform

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "models" / "best_model.pt"
META_PATH = BASE_DIR / "output" / "models" / "meta.json"

# Carrega metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

CLASS_NAMES = meta["class_names"]
IMG_SIZE = meta["img_size"]
NUM_CLASSES = meta["num_classes"]

# Carrega modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES, device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

eval_transform = get_eval_transform(IMG_SIZE)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Teste Visual CNN</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
        .container { margin-top: 60px; }
        input[type=file] { margin: 20px; }
        img { margin-top: 20px; width: 300px; border-radius: 10px; }
        .result { margin-top: 20px; font-size: 1.3em; }
        .loading { font-size: 1.2em; color: #999; }
        button { padding: 10px 25px; border: none; border-radius: 8px; background: #28a745; color: white; cursor: pointer; }
        button:hover { background: #218838; }
    </style>
    <script>
        function showLoading() {
            document.getElementById("loading").style.display = "block";
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Teste Visual da CNN</h1>
        <form action="/" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <input type="file" name="file" accept="image/*" required><br>
            <button type="submit">Enviar Imagem</button>
        </form>
        <div id="loading" class="loading" style="display:none;">Analisando imagem...</div>
        {% if image_url %}
            <img src="{{ image_url }}">
            <div class="result">{{ result_text }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

app = Flask(__name__, static_folder="uploads")
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = UPLOAD_FOLDER / file.filename
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            tensor = eval_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=-1)[0]

            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item() * 100
            class_name = CLASS_NAMES[class_idx]

            result_text = f"O modelo acredita que e: <b>{class_name}</b> ({confidence:.1f}% de confianca)"
            return render_template_string(
                HTML_PAGE,
                image_url=f"/uploads/{file.filename}",
                result_text=result_text
            )
    return render_template_string(HTML_PAGE)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_static_file(filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
