"""
detector.py
Finds vehicles in a photo with a COCO-pretrained Faster R-CNN (no training needed)
and picks the one the user cares about: the vehicle that contains the detected
damage, or the largest vehicle if no damage was found.

This replaces cropping around the damage boxes, which cut off most of the car
and made identification much harder.

Call load_model() once, then detect_vehicles() / select_target_vehicle() / crop_vehicle().
"""

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

# COCO category ids used by torchvision detection models
VEHICLE_LABELS = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

_model = None
_device = None


def load_model():
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    _model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    _model = _model.to(_device).eval()
    return _model


def detect_vehicles(image: Image.Image, score_threshold=0.5):
    """Returns [{label, confidence, bbox{x1,y1,x2,y2}}] for every vehicle in the image."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img_tensor = F.to_tensor(image.convert("RGB")).to(_device)
    with torch.no_grad():
        pred = _model([img_tensor])[0]

    vehicles = []
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score >= score_threshold and label.item() in VEHICLE_LABELS:
            x1, y1, x2, y2 = box.tolist()
            vehicles.append({
                "label": VEHICLE_LABELS[label.item()],
                "confidence": round(score.item(), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
    return vehicles


def _area(b):
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])


def _intersection(a, b):
    return _area({"x1": max(a["x1"], b["x1"]), "y1": max(a["y1"], b["y1"]),
                  "x2": min(a["x2"], b["x2"]), "y2": min(a["y2"], b["y2"])})


def select_target_vehicle(vehicles, damage_boxes=None):
    """
    Picks the vehicle the photo is about.
    - With damage detections: the vehicle that contains the most damage area
      (fraction of each damage box inside it), ties broken by size.
    - Otherwise: the largest vehicle (people photograph the damaged car up close).
    Returns None if no vehicle was found.
    """
    if not vehicles:
        return None

    def damage_overlap(v):
        if not damage_boxes:
            return 0.0
        return sum(_intersection(v["bbox"], d["bbox"]) / max(_area(d["bbox"]), 1e-6)
                   for d in damage_boxes)

    return max(vehicles, key=lambda v: (round(damage_overlap(v), 2), _area(v["bbox"])))


def crop_vehicle(image: Image.Image, vehicle, padding_fraction=0.05):
    """Crops to the vehicle's box plus a little padding. Returns the full image if vehicle is None."""
    if vehicle is None:
        return image
    b = vehicle["bbox"]
    pad_x = (b["x2"] - b["x1"]) * padding_fraction
    pad_y = (b["y2"] - b["y1"]) * padding_fraction
    w, h = image.size
    return image.crop((max(0, b["x1"] - pad_x), max(0, b["y1"] - pad_y),
                       min(w, b["x2"] + pad_x), min(h, b["y2"] + pad_y)))
