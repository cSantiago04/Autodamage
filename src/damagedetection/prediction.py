"""
prediction.py
Prediction for a single image - damage detection
Call load_model() once at startup, then predict_damage() per request.
"""

import json
import os

import torch
from PIL import Image
import torchvision.transforms.functional as F

from src.damagedetection.train import build_model


_model = None
_category_names = None
_device = None


# Original CarDD checkpoint predates <checkpoint>.classes.json
CARDD_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
DEFAULT_CHECKPOINT = "models/damage_detector/maskrcnn_cardd_best.pth"


def class_names_for(checkpoint_path):
    path = os.path.splitext(checkpoint_path)[0] + ".classes.json"
    return json.load(open(path)) if os.path.exists(path) else CARDD_CLASSES


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, nms_thresh=0.3):
    global _model, _category_names, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = class_names_for(checkpoint_path)
    _category_names = {i + 1: n for i, n in enumerate(classes)}

    _model = build_model(len(classes) + 1)
    _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    _model = _model.to(_device)
    _model.roi_heads.nms_thresh = nms_thresh
    _model.eval()

    return _model


def predict_damage(image: Image.Image, confidence_threshold=0.5, return_masks=False):
    """
    Given a PIL Image, returns detected damage instances above the confidence threshold.
    With return_masks=True each result also has "mask": a boolean numpy array (H, W),
    needed to work out which car part the damage is on.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img_tensor = F.to_tensor(image.convert("RGB")).to(_device)

    with torch.no_grad():
        prediction = _model([img_tensor])[0]

    keep = prediction['scores'] > confidence_threshold
    masks = (prediction['masks'][keep][:, 0] > 0.5).cpu().numpy() if return_masks else None

    results = []
    for i, (box, label, score) in enumerate(zip(prediction['boxes'][keep], prediction['labels'][keep],
                                                prediction['scores'][keep])):
        x1, y1, x2, y2 = box.tolist()
        r = {
            "damage_type": _category_names[label.item()],
            "confidence": round(score.item(), 4),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)}
        }
        if return_masks:
            r["mask"] = masks[i]
        results.append(r)

    return results