"""
convert_carparts.py
Converts the Ultralytics Carparts-seg dataset (YOLO polygon .txt labels) into the
same COCO layout as CarDD, so the existing Mask R-CNN data loader and training
code can train a car-part segmentation model unchanged.

The source labels mix granularities (e.g. "front_door" and "front_left_door"), so
left/right variants are merged into 12 consistent part classes. Side is better
inferred later from where the part sits on the detected vehicle.

Usage (from repo root, after downloading Carparts-seg into data/carpartsseg):
    python -m src.damagedetection.convert_carparts
Output: data/carpartsseg/annotations/instances_{train,val,test}2017.json
"""

import json
import os

from PIL import Image

SRC_ROOT = "data/carpartsseg"

SOURCE_NAMES = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light", "back_light",
    "back_right_door", "back_right_light", "front_bumper", "front_door", "front_glass",
    "front_left_door", "front_left_light", "front_light", "front_right_door", "front_right_light",
    "hood", "left_mirror", "object", "right_mirror", "tailgate", "trunk", "wheel",
]

# Canonical part classes (ids must be contiguous from 1; 0 is background)
PART_CLASSES = ["front_bumper", "back_bumper", "hood", "trunk", "front_door", "back_door",
                "front_light", "back_light", "front_glass", "back_glass", "mirror", "wheel"]

MERGE = {
    "front_bumper": "front_bumper", "back_bumper": "back_bumper", "hood": "hood",
    "trunk": "trunk", "tailgate": "trunk",
    "front_door": "front_door", "front_left_door": "front_door", "front_right_door": "front_door",
    "back_door": "back_door", "back_left_door": "back_door", "back_right_door": "back_door",
    "front_light": "front_light", "front_left_light": "front_light", "front_right_light": "front_light",
    "back_light": "back_light", "back_left_light": "back_light", "back_right_light": "back_light",
    "front_glass": "front_glass", "back_glass": "back_glass",
    "left_mirror": "mirror", "right_mirror": "mirror",
    "wheel": "wheel",
    # "object" is dropped (no instances in the release)
}


def polygon_area(xs, ys):
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def convert_split(split):
    img_dir = os.path.join(SRC_ROOT, "images", split)
    lbl_dir = os.path.join(SRC_ROOT, "labels", split)
    cat_id = {name: i + 1 for i, name in enumerate(PART_CLASSES)}

    images, annotations = [], []
    ann_id = 1
    for img_id, fname in enumerate(sorted(os.listdir(img_dir)), start=1):
        w, h = Image.open(os.path.join(img_dir, fname)).size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path) as f:
            for line in f:
                vals = line.split()
                if len(vals) < 7:  # need at least 3 points
                    continue
                part = MERGE.get(SOURCE_NAMES[int(vals[0])])
                if part is None:
                    continue
                xs = [float(v) * w for v in vals[1::2]]
                ys = [float(v) * h for v in vals[2::2]]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id[part],
                    "segmentation": [[c for xy in zip(xs, ys) for c in xy]],
                    "area": polygon_area(xs, ys),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for n, i in cat_id.items()]}
    out_dir = os.path.join(SRC_ROOT, "annotations")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"instances_{split}2017.json")
    with open(out, "w") as f:
        json.dump(coco, f)
    print(f"{split}: {len(images)} images, {len(annotations)} part instances -> {out}")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        convert_split(split)
