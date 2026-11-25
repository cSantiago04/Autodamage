"""
convert_hitl.py
Converts the Humans in the Loop "Car Parts and Car Damages" dataset (Supervisely
format, CC0) into COCO, in the layout the existing data loader / training code uses.

Note: in the download the folder names are swapped:
    "Car damages dataset" holds the PART labels   (998 images, 21 part classes)
    "Car parts dataset"   holds the DAMAGE labels (814 images, 8 damage classes)
441 images appear in both. Those are kept for the test split, so damage-to-part matching
can be measured against real part + damage labels on photos the model never trained on.

Parts are mapped to 18 classes (front/back windows merged into side_window, front/back
wheels into wheel, license plate dropped).

Usage (from repo root):
    python -m src.damagedetection.convert_hitl
    python -m src.damagedetection.convert_hitl --src "data/Car parts and car damages dataset"
Output: <src>/annotations/instances_{train,val,test}2017.json (parts) and
        <src>/annotations/damage_instances_test2017.json (damages on the test images)
"""

import argparse
import glob
import json
import os
import random

PART_DIR = "Car damages dataset/File1"     # swapped names in the release, see above
DAMAGE_DIR = "Car parts dataset/File1"

PART_CLASSES = ["front_bumper", "back_bumper", "hood", "trunk", "front_door", "back_door", "fender",
                "quarter_panel", "rocker_panel", "roof", "headlight", "taillight", "windshield",
                "back_glass", "side_window", "mirror", "wheel", "grille"]
PART_MAP = {
    "Front-bumper": "front_bumper", "Back-bumper": "back_bumper", "Hood": "hood", "Trunk": "trunk",
    "Front-door": "front_door", "Back-door": "back_door", "Fender": "fender",
    "Quarter-panel": "quarter_panel", "Rocker-panel": "rocker_panel", "Roof": "roof",
    "Headlight": "headlight", "Tail-light": "taillight", "Windshield": "windshield",
    "Back-windshield": "back_glass", "Front-window": "side_window", "Back-window": "side_window",
    "Mirror": "mirror", "Front-wheel": "wheel", "Back-wheel": "wheel", "Grille": "grille",
    # "License-plate": dropped (not a repair item)
}
DAMAGE_CLASSES = ["scratch", "dent", "cracked", "paint chip", "flaking", "corrosion", "broken part", "missing part"]


def polygon_area(pts):
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def load_supervisely(folder):
    """{image file name: (width, height, [(class title, exterior points), ...])}"""
    out = {}
    for ann_path in glob.glob(os.path.join(folder, "ann", "*.json")):
        img_name = os.path.basename(ann_path)[:-5]  # "x.jpg.json" -> "x.jpg"
        a = json.load(open(ann_path, encoding="utf-8"))
        objs = [(o["classTitle"], o["points"]["exterior"]) for o in a["objects"]
                if o.get("geometryType") == "polygon" and len(o["points"]["exterior"]) >= 3]
        out[img_name] = (a["size"]["width"], a["size"]["height"], objs)
    return out


def to_coco(items, class_names, class_map, img_prefix=""):
    cat_id = {n: i + 1 for i, n in enumerate(class_names)}
    images, anns = [], []
    for img_id, (name, (w, h, objs)) in enumerate(sorted(items.items()), start=1):
        images.append({"id": img_id, "file_name": img_prefix + name, "width": w, "height": h})
        for title, pts in objs:
            cls = class_map.get(title)
            if cls is None:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
                continue
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_id[cls],
                         "segmentation": [[c for p in pts for c in p]], "area": polygon_area(pts),
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "iscrowd": 0})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for n, i in cat_id.items()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/Car parts and car damages dataset")
    parser.add_argument("--n-test", type=int, default=120, help="test images, drawn from the 441 double-labeled")
    parser.add_argument("--n-val", type=int, default=100)
    args = parser.parse_args()

    parts = load_supervisely(os.path.join(args.src, PART_DIR))
    damages = load_supervisely(os.path.join(args.src, DAMAGE_DIR))
    both = sorted(set(parts) & set(damages))
    rng = random.Random(0)

    test = set(rng.sample(both, min(args.n_test, len(both))))
    rest = sorted(set(parts) - test)
    val = set(rng.sample(rest, args.n_val))
    train = set(rest) - val

    out_dir = os.path.join(args.src, "annotations")
    os.makedirs(out_dir, exist_ok=True)
    for split, names in (("train", train), ("val", val), ("test", test)):
        coco = to_coco({n: parts[n] for n in names}, PART_CLASSES, PART_MAP)
        json.dump(coco, open(os.path.join(out_dir, f"instances_{split}2017.json"), "w"))
        print(f"parts {split}: {len(coco['images'])} images, {len(coco['annotations'])} part instances")

    # Damage labels for the test images (same file names; images live in the damage folder too)
    dmg_map = {"Scratch": "scratch", "Dent": "dent", "Cracked": "cracked", "Paint chip": "paint chip",
               "Flaking": "flaking", "Corrosion": "corrosion", "Broken part": "broken part",
               "Missing part": "missing part"}
    coco = to_coco({n: damages[n] for n in test}, DAMAGE_CLASSES, dmg_map)
    json.dump(coco, open(os.path.join(out_dir, "damage_instances_test2017.json"), "w"))
    print(f"damages on test images: {len(coco['annotations'])} instances")
    print(f"\nImage folder for training: '{PART_DIR}/img'  (use --img-dir-pattern \"{PART_DIR}/img\")")


if __name__ == "__main__":
    main()
