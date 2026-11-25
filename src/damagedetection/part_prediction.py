"""
part_prediction.py
Car-part segmentation for a single image (Mask R-CNN trained on Carparts-seg).
Call load_model() once, then predict_parts() per image.
"""

import json
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.damagedetection.convert_carparts import PART_CLASSES
from src.damagedetection.train import build_model

_model = None
_device = None
_names = None

DEFAULT_CHECKPOINT = "models/part_detector/maskrcnn_carparts_best.pth"


def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """Class names come from <checkpoint>.classes.json (written by train.py); falls back to
    the Carparts-seg classes for checkpoints trained before that file existed."""
    global _model, _device, _names
    classes_path = os.path.splitext(checkpoint_path)[0] + ".classes.json"
    classes = json.load(open(classes_path)) if os.path.exists(classes_path) else PART_CLASSES
    _names = {i + 1: n for i, n in enumerate(classes)}
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(num_classes=len(classes) + 1)
    _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    _model = _model.to(_device).eval()
    return _model


def predict_parts(image: Image.Image, confidence_threshold=0.5):
    """Returns [{part, confidence, bbox, mask (bool HxW numpy)}] for each detected car part."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img_tensor = F.to_tensor(image.convert("RGB")).to(_device)
    with torch.no_grad():
        pred = _model([img_tensor])[0]

    keep = pred["scores"] > confidence_threshold
    masks = (pred["masks"][keep][:, 0] > 0.5).cpu().numpy()
    results = []
    for i, (box, label, score) in enumerate(zip(pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep])):
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "part": _names[label.item()],
            "confidence": round(score.item(), 4),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "mask": masks[i],
        })
    return results
