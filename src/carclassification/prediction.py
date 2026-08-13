"""
prediction.py
Prediction for a single image - car classification
Call load_model() once, then predict_car() per request
"""

import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from src.carclassification.train import build_model
from src.carclassification.data_loader import load_class_names

_model = None
_class_names = None 
_device = None


def load_model(checkpoint_path="models/classifier/resnet50_stanford_cars_best.pth",
               devkit_path="data/carclassificationdataset1/car_devkit/devkit"):
    global _model, _class_names, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _class_names = load_class_names(devkit_path)

    _model = build_model(num_classes=len(_class_names))
    _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    _model = _model.to(_device)
    _model.eval()

    return _model


def predict_car(image: Image.Image, top_k=3):
    """
    Given a PIL Image (assumed to already be cropped/framed appropriately
    by the caller), returns the top-k predicted car classes.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    return [
        {"class_name": _class_names[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]