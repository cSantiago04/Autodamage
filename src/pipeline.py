"""
pipeline.py
Orchestrates the full analysis pipeline: damage detection first (to find
where the actual damaged car is in the frame), then car classification
on a crop derived from the damage location — more reliable than a
generic car detector in cluttered, multi-vehicle photos.
"""

from PIL import Image

from src.carclassification.prediction import load_model as load_classifier, predict_car
from src.damagedetection.prediction import load_model as load_damage_model, predict_damage


def load_models():
    """Call once at API startup."""
    load_classifier()
    load_damage_model()


def crop_to_damaged_car(image: Image.Image, damage_boxes: list, padding_fraction=1.5):
    """
    Crops to the region around detected damage, expanded generously to
    include the surrounding car body. Falls back to the original image
    if no damage was detected.
    """
    if not damage_boxes:
        return image

    x1 = min(d['bbox']['x1'] for d in damage_boxes)
    y1 = min(d['bbox']['y1'] for d in damage_boxes)
    x2 = max(d['bbox']['x2'] for d in damage_boxes)
    y2 = max(d['bbox']['y2'] for d in damage_boxes)

    box_w, box_h = x2 - x1, y2 - y1
    pad_x = box_w * padding_fraction
    pad_y = box_h * padding_fraction

    width, height = image.size
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))


def analyze_image(image: Image.Image):
    """
    Full pipeline: detect damage, crop to the damaged car using that
    location, then classify the car. Returns combined results.
    """
    damage_results = predict_damage(image)
    cropped_image = crop_to_damaged_car(image, damage_results)
    car_results = predict_car(cropped_image)

    return {
        "car": car_results,
        "damage": damage_results,
    }



"""TEMP"""
def analyze_image(image: Image.Image):
    damage_results = predict_damage(image)
    print(f"Original image size: {image.size}")
    cropped_image = crop_to_damaged_car(image, damage_results)
    print(f"Cropped image size: {cropped_image.size}")
    cropped_image.save("debug_cropped.jpg")
    car_results = predict_car(cropped_image)

    return {
        "car": car_results,
        "damage": damage_results,
    }