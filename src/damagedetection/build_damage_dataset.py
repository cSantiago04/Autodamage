"""
build_damage_dataset.py
Builds one COCO damage dataset from three sources, with a unified label set:

  1. CarDD (its own train/val/test splits)
  2. Humans in the Loop damage labels (814 photos of damaged cars). Its test split is the
     same 120 photos held out for the part model, so the two can be evaluated together
  3. Undamaged Carparts-seg photos as negatives (no annotations), so the detector learns
     what intact glass / panels look like instead of finding damage everywhere

Unified classes (repair implied in brackets):
    scratch       <- CarDD scratch, HITL scratch                       [refinish]
    paint damage  <- HITL paint chip, flaking, corrosion               [refinish]
    dent          <- CarDD dent, HITL dent                             [size-dependent]
    crack         <- CarDD crack, HITL cracked                         [repair / replace]
    broken        <- CarDD lamp broken + glass shatter, HITL broken part  [replace]
    missing part  <- HITL missing part                                 [replace]
    tire flat     <- CarDD tire flat                                   [plug / replace]
"Lamp broken" vs "glass shatter" isn't lost: the part model says which part broke.

All file names are relative to data/, so the training data root is data/damage_unified
with --img-dir-pattern "..".

Usage (from repo root):
    python -m src.damagedetection.build_damage_dataset
Output: data/damage_unified/annotations/instances_{train,val,test,test_cardd,test_hitl,test_negatives}2017.json
"""

import json
import os
import random
import re

from src.damagedetection.convert_hitl import DAMAGE_DIR, load_supervisely, polygon_area

DATA = "data"
OUT_ROOT = os.path.join(DATA, "damage_unified")
CARDD = "CarDD_release/CarDD_release/CarDD_COCO"
HITL = "Car parts and car damages dataset"
CARPARTS = "carpartsseg"

CLASSES = ["scratch", "paint damage", "dent", "crack", "broken", "missing part", "tire flat"]
CARDD_MAP = {"scratch": "scratch", "dent": "dent", "crack": "crack", "lamp broken": "broken",
             "glass shatter": "broken", "tire flat": "tire flat"}
HITL_MAP = {"Scratch": "scratch", "Paint chip": "paint damage", "Flaking": "paint damage",
            "Corrosion": "paint damage", "Dent": "dent", "Cracked": "crack", "Broken part": "broken",
            "Missing part": "missing part"}

# Model / old-checkpoint label names -> unified names (used when comparing the old CarDD model)
TO_UNIFIED = {**CARDD_MAP, **{v: v for v in CLASSES}}


class Builder:
    def __init__(self):
        self.images, self.anns = [], []
        self.cat_id = {n: i + 1 for i, n in enumerate(CLASSES)}

    def add_image(self, rel_path, w, h, source):
        img_id = len(self.images) + 1
        self.images.append({"id": img_id, "file_name": rel_path.replace("\\", "/"), "width": w, "height": h,
                            "source": source})
        return img_id

    def add_ann(self, img_id, cls, segmentation, bbox, area):
        self.anns.append({"id": len(self.anns) + 1, "image_id": img_id, "category_id": self.cat_id[cls],
                          "segmentation": segmentation, "bbox": bbox, "area": area, "iscrowd": 0})

    def coco(self, keep=None):
        imgs = [i for i in self.images if keep is None or keep(i)]
        ids = {i["id"] for i in imgs}
        return {"images": imgs, "annotations": [a for a in self.anns if a["image_id"] in ids],
                "categories": [{"id": i, "name": n} for n, i in self.cat_id.items()]}


def add_cardd(b, split):
    coco = json.load(open(os.path.join(DATA, CARDD, "annotations", f"instances_{split}2017.json")))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        new_id = b.add_image(f"{CARDD}/{split}2017/{im['file_name']}", im["width"], im["height"], "cardd")
        for a in by_img.get(im["id"], []):
            b.add_ann(new_id, CARDD_MAP[names[a["category_id"]]], a["segmentation"], a["bbox"], a["area"])


def add_hitl(b, items):
    for name, (w, h, objs) in sorted(items.items()):
        img_id = b.add_image(f"{HITL}/{DAMAGE_DIR}/img/{name}", w, h, "hitl")
        for title, pts in objs:
            cls = HITL_MAP.get(title)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if cls is None or max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
                continue
            b.add_ann(img_id, cls, [[c for p in pts for c in p]],
                      [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], polygon_area(pts))


def add_negatives(b, split, n, rng):
    """Undamaged Carparts-seg photos, one per source image (Roboflow saved ~5 augmented copies each)."""
    from PIL import Image
    folder = os.path.join(DATA, CARPARTS, "images", split)
    by_source = {}
    for f in sorted(os.listdir(folder)):
        by_source.setdefault(re.sub(r"\.rf\..*$", "", f), f)
    for f in rng.sample(sorted(by_source.values()), min(n, len(by_source))):
        w, h = Image.open(os.path.join(folder, f)).size
        b.add_image(f"{CARPARTS}/images/{split}/{f}", w, h, "negative")


def main():
    rng = random.Random(0)
    hitl = load_supervisely(os.path.join(DATA, HITL, DAMAGE_DIR))
    part_test = json.load(open(os.path.join(DATA, HITL, "annotations", "instances_test2017.json")))
    hitl_test = {i["file_name"] for i in part_test["images"]}  # same 120 held-out photos as the part model
    rest = sorted(set(hitl) - hitl_test)
    hitl_val = set(rng.sample(rest, 70))
    hitl_train = set(rest) - hitl_val

    os.makedirs(os.path.join(OUT_ROOT, "annotations"), exist_ok=True)
    for split, hitl_names, n_neg in (("train", hitl_train, 400), ("val", hitl_val, 50), ("test", hitl_test, 60)):
        b = Builder()
        add_cardd(b, split)
        add_hitl(b, {n: hitl[n] for n in hitl_names if n in hitl})
        add_negatives(b, split, n_neg, rng)
        outs = {split: b.coco()}
        if split == "test":
            for src, key in (("cardd", "test_cardd"), ("hitl", "test_hitl"), ("negative", "test_negatives")):
                outs[key] = b.coco(lambda i, s=src: i["source"] == s)
        for key, coco in outs.items():
            json.dump(coco, open(os.path.join(OUT_ROOT, "annotations", f"instances_{key}2017.json"), "w"))
            by_src = {}
            for i in coco["images"]:
                by_src[i["source"]] = by_src.get(i["source"], 0) + 1
            print(f"{key:<15} {len(coco['images']):>5} images {by_src}, {len(coco['annotations'])} damages")


if __name__ == "__main__":
    main()
