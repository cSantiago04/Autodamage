"""
describe_damage.py
Prototype of the hybrid pipeline: Mask R-CNN localizes damage, then Gemma looks at
the full image (plus those detections as hints) and returns the vehicle, the
damaged parts, and a severity for each, i.e. the (part + severity) labels the
cost stage needs and CarDD doesn't have.

Usage (from repo root):
    python -m src.gemma.describe_damage "data/Test Images/Test3.jpg"
    python -m src.gemma.describe_damage "data/Test Images" --no-detector
"""

import argparse
import json
import os

from PIL import Image

from src.gemma.client import DEFAULT_MODEL, ask_json, check_server

SEVERITIES = ["minor", "moderate", "severe"]
REPAIR_ACTIONS = ["paintless dent repair", "repair and refinish", "replace", "refinish only", "unknown"]

SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle": {
            "type": "object",
            "properties": {
                "make": {"type": "string"},
                "model": {"type": "string"},
                "year_range": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["make", "model", "year_range", "confidence"],
        },
        "damages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string"},
                    "damage_type": {"type": "string"},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "likely_repair": {"type": "string", "enum": REPAIR_ACTIONS},
                    "description": {"type": "string"},
                },
                "required": ["part", "damage_type", "severity", "likely_repair", "description"],
            },
        },
    },
    "required": ["vehicle", "damages"],
}


def build_prompt(detections):
    prompt = (
        "You are an auto body damage assessor. Look at this photo of a vehicle.\n"
        "1. Identify the vehicle (make, model, approximate year range).\n"
        "2. List every damaged exterior part (e.g. 'front bumper cover', 'driver front door', "
        "'left headlight assembly'). For each, give the damage type, severity "
        "(minor / moderate / severe), and the most likely repair action.\n"
        "Only report damage you can actually see."
    )
    if detections:
        hints = "\n".join(
            f"- {d['damage_type']} (conf {d['confidence']:.2f}) at box "
            f"x1={d['bbox']['x1']:.0f}, y1={d['bbox']['y1']:.0f}, x2={d['bbox']['x2']:.0f}, y2={d['bbox']['y2']:.0f}"
            for d in detections
        )
        prompt += ("\n\nA damage detector flagged these regions (pixel coords in the original image; "
                   "it can be wrong, use them as hints):\n" + hints)
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="image file or folder of images")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-detector", action="store_true", help="skip Mask R-CNN hints")
    args = parser.parse_args()

    check_server(args.model)

    if not args.no_detector:
        from src.damagedetection.prediction import load_model, predict_damage
        load_model()

    if os.path.isdir(args.path):
        paths = [os.path.join(args.path, f) for f in sorted(os.listdir(args.path))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        paths = [args.path]

    for p in paths:
        img = Image.open(p)
        detections = [] if args.no_detector else predict_damage(img)
        result = ask_json(build_prompt(detections), img, SCHEMA, model=args.model)
        print(f"\n=== {p} ===")
        if detections:
            print(f"Mask R-CNN: {[d['damage_type'] for d in detections]}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
