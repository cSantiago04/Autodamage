"""Quick smoke test of the full pipeline on one image (run from repo root)."""
import json

from PIL import Image

from src.pipeline import analyze_image, load_models

load_models()  # optionally car_identifier="gemma4:e4b" to also suggest the car from the photo

img = Image.open("data/Test Images/Test3.jpg")
result = analyze_image(img)  # pass vin="..." to decode the vehicle
print(json.dumps(result, indent=2, default=str))
