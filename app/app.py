#!/usr/bin/env python3
import os, time, imghdr
from pathlib import Path
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, session

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "ml" / "datasets"

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

app = Flask(__name__)
# A chave secreta é essencial para o funcionamento das sessões
app.secret_key = "change-this-in-production-to-a-secure-random-key"
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Autenticação ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Credenciais fixas conforme solicitado
        if username == "Yöttun" and password == "admin":
            session['logged_in'] = True
            flash("Login realizado com sucesso!", "success")
            return redirect(url_for("index"))
        else:
            flash("Usuário ou senha inválidos.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Você foi desconectado.", "info")
    return redirect(url_for('login'))

# --- Funções Auxiliares (sem mudanças) ---
def sanitize_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    return "".join(c for c in name if c.isalnum() or c in ("-", "_"))

def list_converters():
    return sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])

def is_allowed_file(filename: str) -> bool:
    if "." not in filename: return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def guess_ext_by_content(filepath: Path) -> str | None:
    kind = imghdr.what(filepath)
    if not kind: return None
    if kind == "jpeg": return "jpg"
    return kind

def save_image(converter: str, storage_file) -> str:
    folder = DATASET_DIR / converter
    folder.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{converter}_{ts}_{int(time.time()*1000)%1000:03d}"
    tmp_path = folder / f"{base}.uploading"
    storage_file.save(str(tmp_path))
    real_ext = guess_ext_by_content(tmp_path)
    if real_ext is None:
        if is_allowed_file(storage_file.filename):
            real_ext = storage_file.filename.rsplit(".", 1)[1].lower()
        else:
            tmp_path.unlink(missing_ok=True)
            raise ValueError("Arquivo enviado não parece ser uma imagem válida.")
    final_name = f"{base}.{real_ext}"
    tmp_path.rename(folder / final_name)
    return final_name

def list_images(converter: str):
    folder = DATASET_DIR / converter
    if not folder.exists(): return []
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower().lstrip(".") in ALLOWED_EXT]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{
        "name": f.name,
        "url": url_for("serve_image", converter=converter, filename=f.name),
        "size_kb": max(1, f.stat().st_size // 1024),
        "count": len(files)
    } for f in files]


# --- Rotas da Aplicação (agora protegidas) ---

@app.route("/")
@login_required
def index():
    converters = list_converters()
    return render_template("index.html", converters=converters)

@app.route("/add-converter", methods=["POST"])
@login_required
def add_converter():
    conv = sanitize_name(request.form.get("converter_name", ""))
    if not conv:
        flash("Nome inválido.", "error"); return redirect(url_for("index"))
    (DATASET_DIR / conv).mkdir(parents=True, exist_ok=True)
    flash(f"Conversor “{conv}” criado.", "success")
    return redirect(url_for("index"))

# Rota /upload foi removida pois /upload-batch faz tudo agora.

@app.route("/upload-batch", methods=["POST"])
@login_required
def upload_batch():
    conv = sanitize_name(request.form.get("converter_select_batch", ""))
    if not conv:
        flash("Selecione um conversor.", "error"); return redirect(url_for("index"))
    files = request.files.getlist("photos")
    if not files:
        flash("Nenhuma imagem na fila.", "info"); return redirect(url_for("index"))
    saved = 0; failed = 0
    for f in files:
        if f and f.filename:
            try:
                if "." in f.filename and not is_allowed_file(f.filename):
                    failed += 1; continue
                save_image(conv, f); saved += 1
            except Exception:
                failed += 1
    msg = f"✅ {saved} foto(s) enviada(s) com sucesso!"
    if failed: msg += f" • ❌ {failed} falharam."
    flash(msg, "success" if saved > 0 and failed == 0 else "info")
    return redirect(url_for("converter_gallery", converter=conv))

@app.route("/converter/<converter>")
@login_required
def converter_gallery(converter: str):
    converter = sanitize_name(converter)
    if not converter or not (DATASET_DIR / converter).exists():
        flash("Conversor inexistente.", "error"); return redirect(url_for("index"))
    images = list_images(converter)
    image_count = len(images)
    return render_template("converter.html", converter=converter, images=images, image_count=image_count)

@app.route("/delete", methods=["POST"])
@login_required
def delete_image():
    converter = sanitize_name(request.form.get("converter", ""))
    filename  = request.form.get("filename", "")
    target = DATASET_DIR / converter / filename
    if target.exists() and target.is_file():
        target.unlink(); flash("Imagem removida.", "info")
    else:
        flash("Arquivo não encontrado.", "error")
    return redirect(url_for("converter_gallery", converter=converter))

@app.route("/datasets/<converter>/<path:filename>")
@login_required
def serve_image(converter: str, filename: str):
    return send_from_directory(DATASET_DIR / sanitize_name(converter), filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=52500, debug=False)
