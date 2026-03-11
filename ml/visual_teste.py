import os
from pathlib import Path
from flask import Flask, render_template_string, request
import tensorflow as tf
import numpy as np
from PIL import Image

# Caminhos base
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "models" / "best_model.keras"

# Carrega modelo
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = model.input_shape[1]
CLASS_NAMES = list(model.output_names) if hasattr(model, "output_names") else None

# Interface simples em HTML
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
        <div id="loading" class="loading" style="display:none;">🔄 Analisando imagem...</div>
        {% if image_url %}
            <img src="{{ image_url }}">
            <div class="result">{{ result_text }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

app = Flask(__name__, static_folder="uploads")
UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = UPLOAD_FOLDER / file.filename
            file.save(image_path)

            # Preprocessa imagem
            img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx] * 100

            # Caso o modelo tenha class_names salvos
            if hasattr(model, "class_names"):
                class_name = model.class_names[class_idx]
            else:
                class_name = f"Classe {class_idx}"

            result_text = f"🧠 O modelo acredita que é: <b>{class_name}</b> ({confidence:.1f}% de confiança)"
            return render_template_string(
                HTML_PAGE,
                image_url=f"/uploads/{file.filename}",
                result_text=result_text
            )
    return render_template_string(HTML_PAGE)

# Servir as imagens enviadas
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
