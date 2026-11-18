from PIL import Image
from src.pipeline import load_models, analyze_image

load_models()

img = Image.open("data/Test images/Test3.jpg")

result = analyze_image(img)
print(result)