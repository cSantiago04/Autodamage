"""
client.py
Minimal client for running Gemma (or any vision model) through a local Ollama server.

Setup (one time):
    1. Install Ollama: https://ollama.com/download  (or: winget install Ollama.Ollama)
    2. Pull a vision-capable Gemma model:  ollama pull gemma4:e4b
    3. Ollama runs a server on http://localhost:11434 automatically after install.
"""

import base64
import io
import json

import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"


def encode_image(image: Image.Image, max_side=1024) -> str:
    """Downscale (keeping aspect ratio) and base64-encode an image as JPEG."""
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_server(model=DEFAULT_MODEL):
    """Raise a helpful error if Ollama isn't running or the model isn't pulled."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.ConnectionError:
        raise RuntimeError("Can't reach Ollama at localhost:11434. Is it installed and running?")
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(f"Model '{model}' not found locally. Run: ollama pull {model}\n"
                           f"Available: {sorted(names) or 'none'}")


def ask_json(prompt: str, image: Image.Image, schema: dict, model=DEFAULT_MODEL, timeout=300) -> dict:
    """
    Sends an image + prompt to the model and returns parsed JSON.
    `schema` is a JSON schema; Ollama constrains the output to match it.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [encode_image(image)]}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])
