"""
evaluate_repair_action.py
Sanity-checks the repair-action rules against the hand severity labels
(labels/severity_labels.csv). Actions map onto the labels by amount of work:
    refinish only / paintless dent repair ~ minor
    repair and refinish / tire repair     ~ moderate
    replace                               ~ severe

For each labeled CarDD damage it uses the ground-truth damage mask (so this checks the
rules, not the damage detector), runs the part segmenter, matches the damage to a part,
picks an action, and compares.

Usage (from repo root):
    python -m src.damagedetection.evaluate_repair_action
    python -m src.damagedetection.evaluate_repair_action --search   # also grid-search extent cutoffs
"""

import argparse
import contextlib
import csv
import io
import itertools
import os
from collections import Counter

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from src.damagedetection import part_prediction
from src.damagedetection.repair_action import (ACTION_LEVEL, EXTENT_RULES, PDR, REFINISH, REPAIR, REPLACE,
                                               assign_part, choose_action, collision_context)

DATA_ROOT = "data/CarDD_release/CarDD_release/CarDD_COCO"
LABELS = "labels/severity_labels.csv"
LEVELS = ["minor", "moderate", "severe"]


def collect_features(split="test"):
    """Per labeled damage: type, label level, matched part, extent, image fraction."""
    with open(LABELS, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["severity"] in LEVELS]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(os.path.join(DATA_ROOT, "annotations", f"instances_{split}2017.json"))

    part_prediction.load_model()
    parts_cache, feats = {}, []
    for r in tqdm(rows, desc="Matching damage to parts"):
        ann = coco.loadAnns(int(r["ann_id"]))[0]
        if ann["image_id"] not in parts_cache:
            img = Image.open(os.path.join(DATA_ROOT, f"{split}2017", r["file_name"]))
            parts_cache[ann["image_id"]] = part_prediction.predict_parts(img)
        mask = coco.annToMask(ann).astype(bool)
        match = assign_part(mask, parts_cache[ann["image_id"]])
        image_types = [coco.loadCats(a["category_id"])[0]["name"]
                       for a in coco.loadAnns(coco.getAnnIds(imgIds=ann["image_id"]))]
        feats.append({"type": r["damage_type"], "label": LEVELS.index(r["severity"]), "part": match["part"],
                      "extent": match["extent"], "image_fraction": mask.sum() / mask.size,
                      "collision": collision_context(image_types)})
    return feats


def predict(feats, extent_rules=None, use_collision=True):
    return [choose_action(f["type"], f["part"], f["extent"], f["image_fraction"], extent_rules,
                          collision=use_collision and f["collision"]) for f in feats]


def accuracy(feats, preds):
    return np.mean([ACTION_LEVEL[a] == f["label"] for (a, _), f in zip(preds, feats)])


def report(feats, preds, title):
    print(f"\n===== {title} =====")
    print(f"level agreement: {accuracy(feats, preds):.1%}   (n={len(feats)})")
    print("\nrows = hand label, cols = rule level (minor ~ refinish/PDR, moderate ~ repair, severe ~ replace):")
    print(f"{'':>10}" + "".join(f"{s:>10}" for s in LEVELS))
    for li, l in enumerate(LEVELS):
        print(f"{l:>10}" + "".join(
            f"{sum(1 for (a, _), f in zip(preds, feats) if f['label'] == li and ACTION_LEVEL[a] == pi):>10}"
            for pi in range(3)))
    print("\nper damage type:")
    for t in sorted({f["type"] for f in feats}):
        idx = [i for i, f in enumerate(feats) if f["type"] == t]
        print(f"  {t:<14} {accuracy([feats[i] for i in idx], [preds[i] for i in idx]):.0%}  (n={len(idx)})")
    print("\ndecided by:", dict(Counter(b for _, b in preds)))


def grid_search(feats):
    """Best extent cutoffs per size-dependent type. Small n and tuned on the same labels: a hint, not truth."""
    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    shapes = {  # same action sequence as EXTENT_RULES, only the cutoffs move
        "dent": lambda c: [(c[0], PDR), (c[1], REPAIR), (None, REPLACE)],
        "scratch": lambda c: [(c[0], REFINISH), (None, REPAIR)],
        "crack": lambda c: [(c[0], REPAIR), (None, REPLACE)],
    }
    best = dict(EXTENT_RULES)
    for t, shape in shapes.items():
        sub = [f for f in feats if f["type"] == t and f["extent"] is not None]
        if len(sub) < 5:
            continue
        n_cut = len(EXTENT_RULES[t]) - 1
        cands = list(itertools.combinations(grid, n_cut))
        best[t] = shape(max(cands, key=lambda c: accuracy(sub, predict(sub, {**best, t: shape(c)}))))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="grid-search extent cutoffs")
    args = parser.parse_args()

    feats = collect_features()
    print(f"\nPart matched for {sum(f['part'] is not None for f in feats)}/{len(feats)} damages "
          f"(rest fall back to image extent)")
    print("Matched parts:", dict(Counter(f["part"] for f in feats if f["part"])))

    report(feats, predict(feats, use_collision=False), "Extent rules only (no collision context)")
    report(feats, predict(feats), "Current rules (with collision context)")
    if args.search:
        best = grid_search(feats)
        report(feats, predict(feats, best), "Grid-searched cutoffs (optimistic: tuned on the same labels)")
        print("\nSuggested EXTENT_RULES =", best)


if __name__ == "__main__":
    main()
