"""
pipeline.py
Orchestrates the full analysis pipeline:
  1. Damage detection (Mask R-CNN, trained on CarDD)
  2. Vehicle detection (COCO-pretrained Faster R-CNN) -> pick the vehicle that
     contains the damage, crop to it
  3. Keep only damage that lies on that vehicle (drops detections on other cars)
  4. Identify the vehicle from the crop, either with the ResNet-50 Stanford Cars
     classifier or an Ollama vision model (e.g. "gemma4:e4b", "qwen2.5vl:7b")
"""

from PIL import Image

from src.carclassification import vehicle_detector as detector
from src.damagedetection.prediction import load_model as load_damage_model, predict_damage

_car_identifier = None


def load_models(car_identifier="resnet"):
    """
    Call once at API startup.
    car_identifier: "resnet" for the Stanford Cars classifier, or an Ollama model name.
    """
    global _car_identifier
    _car_identifier = car_identifier
    load_damage_model()
    detector.load_model()
    if car_identifier == "resnet":
        from src.carclassification.prediction import load_model as load_classifier
        load_classifier()
    else:
        from src.gemma.client import check_server
        check_server(car_identifier)


def damage_on_vehicle(damage, vehicle, min_overlap=0.5):
    """Keeps damage boxes that are mostly (>= min_overlap of their area) inside the vehicle box."""
    if vehicle is None:
        return damage
    v = vehicle["bbox"]
    kept = []
    for d in damage:
        b = d["bbox"]
        inter = max(0, min(b["x2"], v["x2"]) - max(b["x1"], v["x1"])) * \
                max(0, min(b["y2"], v["y2"]) - max(b["y1"], v["y1"]))
        area = max((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), 1e-6)
        if inter / area >= min_overlap:
            kept.append(d)
    return kept


def identify_vehicle(crop: Image.Image):
    if _car_identifier == "resnet":
        from src.carclassification.prediction import predict_car
        return {"source": "resnet", "top_k": predict_car(crop)}
    from src.gemma.identify_car import identify_car
    return {"source": _car_identifier, **identify_car(crop, model=_car_identifier)}


def analyze_image(image: Image.Image):
    """Full pipeline. Returns the target vehicle box, its identification, and its damage."""
    if _car_identifier is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    damage = predict_damage(image)
    vehicles = detector.detect_vehicles(image)
    target = detector.select_target_vehicle(vehicles, damage)
    crop = detector.crop_vehicle(image, target)

    return {
        "vehicle_found": target is not None,
        "vehicle_box": target,
        "car": identify_vehicle(crop),
        "damage": damage_on_vehicle(damage, target),
    }
