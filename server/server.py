#!/usr/bin/env python3
"""
API REST para coleta de fotos de conversores.
Recebe requisições do APK Android e gerencia as pastas de dataset localmente.
"""

import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "ml" / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

# Magic bytes para detectar tipo de imagem (substitui imghdr que foi removido)
_SIGNATURES = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # WebP começa com RIFF....WEBP
    b"BM": "bmp",
}

# Credenciais fixas
USERS = {"Yottun": "admin"}

app = FastAPI(title="CNN Fotos API")
security = HTTPBasic()


# --- Auth ---

def auth(credentials: HTTPBasicCredentials = Depends(security)):
    if USERS.get(credentials.username) != credentials.password:
        raise HTTPException(401, "Credenciais inválidas")
    return credentials.username


# --- Helpers ---

def sanitize(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    return "".join(c for c in name if c.isalnum() or c in ("-", "_"))


def guess_ext(filepath: Path) -> str | None:
    header = filepath.read_bytes()[:16]
    for magic, ext in _SIGNATURES.items():
        if header.startswith(magic):
            if ext == "webp" and b"WEBP" not in header:
                continue
            return ext
    return None


# --- Endpoints ---

@app.get("/conversores")
def listar_conversores(_user: str = Depends(auth)):
    dirs = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    return {"conversores": dirs}


@app.post("/conversores")
def criar_conversor(nome: str, _user: str = Depends(auth)):
    nome = sanitize(nome)
    if not nome:
        raise HTTPException(400, "Nome inválido")
    pasta = DATASET_DIR / nome
    pasta.mkdir(parents=True, exist_ok=True)
    return {"criado": nome}


@app.get("/conversores/{conversor}/fotos")
def listar_fotos(conversor: str, _user: str = Depends(auth)):
    conversor = sanitize(conversor)
    pasta = DATASET_DIR / conversor
    if not pasta.exists():
        raise HTTPException(404, "Conversor não encontrado")
    fotos = []
    for f in sorted(pasta.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file() and f.suffix.lower().lstrip(".") in ALLOWED_EXT:
            fotos.append({
                "nome": f.name,
                "tamanho_kb": max(1, f.stat().st_size // 1024),
            })
    return {"conversor": conversor, "total": len(fotos), "fotos": fotos}


@app.post("/conversores/{conversor}/fotos")
async def enviar_fotos(conversor: str, fotos: list[UploadFile], _user: str = Depends(auth)):
    conversor = sanitize(conversor)
    if not conversor:
        raise HTTPException(400, "Conversor inválido")
    pasta = DATASET_DIR / conversor
    pasta.mkdir(parents=True, exist_ok=True)

    salvos = 0
    falhas = 0

    for foto in fotos:
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            base = f"{conversor}_{ts}_{int(time.time()*1000)%1000:03d}"
            tmp = pasta / f"{base}.uploading"

            conteudo = await foto.read()
            tmp.write_bytes(conteudo)

            ext = guess_ext(tmp)
            if ext is None:
                nome_orig = foto.filename or ""
                if "." in nome_orig and nome_orig.rsplit(".", 1)[1].lower() in ALLOWED_EXT:
                    ext = nome_orig.rsplit(".", 1)[1].lower()
                else:
                    tmp.unlink(missing_ok=True)
                    falhas += 1
                    continue

            if ext not in ALLOWED_EXT:
                tmp.unlink(missing_ok=True)
                falhas += 1
                continue

            final = pasta / f"{base}.{ext}"
            tmp.rename(final)
            salvos += 1
        except Exception:
            falhas += 1

    return {"salvos": salvos, "falhas": falhas}


@app.delete("/conversores/{conversor}/fotos/{arquivo}")
def deletar_foto(conversor: str, arquivo: str, _user: str = Depends(auth)):
    conversor = sanitize(conversor)
    caminho = DATASET_DIR / conversor / arquivo
    if not caminho.exists() or not caminho.is_file():
        raise HTTPException(404, "Foto não encontrada")
    caminho.unlink()
    return {"deletado": arquivo}


@app.get("/conversores/{conversor}/fotos/{arquivo}/download")
def baixar_foto(conversor: str, arquivo: str, _user: str = Depends(auth)):
    conversor = sanitize(conversor)
    caminho = DATASET_DIR / conversor / arquivo
    if not caminho.exists() or not caminho.is_file():
        raise HTTPException(404, "Foto não encontrada")
    return FileResponse(caminho)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52500)
