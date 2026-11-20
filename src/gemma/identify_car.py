"""
identify_car.py
Open-set vehicle identification with Gemma (via Ollama).

The model is asked to list visible evidence first (badges, emblems, lettering,
distinctive headlight/grille/taillight shapes, body style) and only then name the
car. Requiring the evidence field before the answer makes it actually read badges
like "Si" or "RSX" instead of guessing from overall shape.

Usage (from repo root):
    python -m src.gemma.identify_car "data/Test Images"            # crops to the vehicle first
    python -m src.gemma.identify_car "data/Test Images" --no-crop
"""

import argparse
import json
import os

from PIL import Image

from src.gemma.client import DEFAULT_MODEL, ask_json, check_server

PROMPT = (
    "You are an automotive expert identifying a vehicle from a photo, which may show a "
    "damaged car, at night, or from an unusual angle.\n"
    "Step 1: List the concrete visual evidence you can see: badges, emblems, model or trim "
    "lettering (read any text exactly), and distinctive grille, headlight, taillight, wheel, "
    "and body-style details.\n"
    "Step 2: Using that evidence, give the make, model, trim (if a trim badge is visible), "
    "body style, and the model-year range of that generation.\n"
    "Prefer what badges say over overall shape. If no vehicle is clearly visible, set "
    "vehicle_present to false."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_present": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "make": {"type": "string"},
        "model": {"type": "string"},
        "trim": {"type": "string"},
        "body_style": {"type": "string", "enum": ["sedan", "coupe", "hatchback", "convertible", "wagon",
                                                  "suv", "pickup", "van", "minivan", "other"]},
        "year_range": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    # evidence is listed before make/model so the model writes it first
    "required": ["vehicle_present", "evidence", "make", "model", "trim", "body_style",
                 "year_range", "confidence"],
}


def identify_car(image: Image.Image, model=DEFAULT_MODEL) -> dict:
    return ask_json(PROMPT, image, SCHEMA, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="image file or folder of images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-crop", action="store_true", help="skip cropping to the target vehicle")
    args = parser.parse_args()

    check_server(args.model)
    if not args.no_crop:
        from src.cardetection import detector
        detector.load_model()

    if os.path.isdir(args.path):
        paths = [os.path.join(args.path, f) for f in sorted(os.listdir(args.path))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        paths = [args.path]

    for p in paths:
        img = Image.open(p)
        if not args.no_crop:
            img = detector.crop_vehicle(img, detector.select_target_vehicle(detector.detect_vehicles(img)))
        print(f"\n=== {p} ===")
        print(json.dumps(identify_car(img, model=args.model), indent=2))


if __name__ == "__main__":
    main()
