"""Quick smoke test of the full pipeline on one image (run from repo root)."""
from PIL import Image
from src.pipeline import load_models, analyze_image

load_models(car_identifier="resnet")  # or an Ollama model, e.g. "qwen2.5vl:7b"

img = Image.open("data/Test Images/Test3.jpg")

result = analyze_image(img)
print(result)
