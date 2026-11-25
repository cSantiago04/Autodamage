"""
pipeline.py
Orchestrates the full analysis pipeline:
  1. Vehicle: decoded from the VIN (NHTSA vPIC) when one is given. Optionally, an
     identifier suggests the car from the photo ("resnet" or an Ollama vision model)
  2. Damage detection (Mask R-CNN, trained on CarDD), with masks
  3. Vehicle detection (COCO-pretrained Faster R-CNN): find the vehicle that contains
     the damage and drop detections on other cars
  4. Part segmentation (Mask R-CNN, trained on Carparts-seg)
  5. Match each damage to its part and choose the repair action (src/damagedetection/repair_action.py)
"""

from PIL import Image

from src.carclassification import vehicle_detector as detector
from src.carclassification.vin_decoder import decode_vin
from src.damagedetection import part_prediction
from src.damagedetection.prediction import load_model as load_damage_model, predict_damage
from src.damagedetection.repair_action import assess

_loaded = False
_car_identifier = None


def load_models(car_identifier=None, part_checkpoint=None):
    """
    Call once at API startup.
    car_identifier: None (VIN only), "resnet" (Stanford Cars classifier), or an Ollama
    model name, used to suggest the car from the photo.
    """
    global _loaded, _car_identifier
    _car_identifier = car_identifier
    load_damage_model()
    detector.load_model()
    part_prediction.load_model(part_checkpoint or part_prediction.DEFAULT_CHECKPOINT)
    if car_identifier == "resnet":
        from src.carclassification.prediction import load_model as load_classifier
        load_classifier()
    elif car_identifier:
        from src.gemma.client import check_server
        check_server(car_identifier)
    _loaded = True


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


def suggest_vehicle(crop: Image.Image):
    if _car_identifier == "resnet":
        from src.carclassification.prediction import predict_car
        return {"source": "resnet", "top_k": predict_car(crop)}
    from src.gemma.identify_car import identify_car
    return {"source": _car_identifier, **identify_car(crop, model=_car_identifier)}


def analyze_image(image: Image.Image, vin=None):
    """
    Full pipeline. Returns the vehicle (from the VIN and/or a photo-based suggestion),
    the target vehicle box, and each damage with its part, extent, and repair action.
    """
    if not _loaded:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    damage = predict_damage(image, return_masks=True)
    vehicles = detector.detect_vehicles(image)
    target = detector.select_target_vehicle(vehicles, damage)
    damage = damage_on_vehicle(damage, target)
    parts = part_prediction.predict_parts(image)

    result = {
        "vehicle": decode_vin(vin) if vin else None,
        "vehicle_suggestion": suggest_vehicle(detector.crop_vehicle(image, target)) if _car_identifier else None,
        "vehicle_found": target is not None,
        "vehicle_box": target,
        "parts_detected": sorted({p["part"] for p in parts}),
        "damage": assess(damage, parts),
    }
    return result
