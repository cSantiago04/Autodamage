"""
evaluate_severity.py
Measures the severity rules against your hand labels (labels/severity_labels.csv,
made with label_severity.py).

For each labeled CarDD damage it uses the ground-truth damage mask (so this measures
the severity logic, not the damage detector), runs the part segmenter on the image,
matches the damage to a part, applies the rules, and compares with your label.

Usage (from repo root):
    python -m src.damagedetection.evaluate_severity
    python -m src.damagedetection.evaluate_severity --search    # also grid-search thresholds
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
from src.damagedetection.severity import (IMAGE_FRACTION_THRESHOLDS, PART_FRACTION_THRESHOLDS,
                                          SEVERITIES, assign_part, rate_severity)

DATA_ROOT = "data/CarDD_release/CarDD_release/CarDD_COCO"
LABELS = "labels/severity_labels.csv"


def collect_features(split="test"):
    """Per labeled damage: type, label, matched part, part_fraction, image_fraction."""
    with open(LABELS, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["severity"] in SEVERITIES]
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
        feats.append({"type": r["damage_type"], "label": r["severity"], "part": match["part"],
                      "part_fraction": match["part_fraction"], "image_fraction": mask.sum() / mask.size})
    return feats


def score(feats, part_t=None, image_t=None):
    preds = [rate_severity(f["type"], f["part_fraction"], f["image_fraction"], part_t, image_t)[0]
             for f in feats]
    labels = [f["label"] for f in feats]
    exact = np.mean([p == l for p, l in zip(preds, labels)])
    within_one = np.mean([abs(SEVERITIES.index(p) - SEVERITIES.index(l)) <= 1 for p, l in zip(preds, labels)])
    return exact, within_one, preds


def report(feats, preds, title):
    labels = [f["label"] for f in feats]
    print(f"\n===== {title} =====")
    print(f"exact accuracy: {np.mean([p == l for p, l in zip(preds, labels)]):.1%}   (n={len(feats)})")
    print("\nconfusion (rows = your label, cols = rules):")
    print(f"{'':>10}" + "".join(f"{s:>10}" for s in SEVERITIES))
    for l in SEVERITIES:
        print(f"{l:>10}" + "".join(f"{sum(1 for p, t in zip(preds, labels) if t == l and p == s):>10}"
                                   for s in SEVERITIES))
    print("\nper damage type:")
    for t in sorted({f["type"] for f in feats}):
        idx = [i for i, f in enumerate(feats) if f["type"] == t]
        acc = np.mean([preds[i] == labels[i] for i in idx])
        print(f"  {t:<14} {acc:.0%}  (n={len(idx)})")


def grid_search(feats):
    """Best part-fraction cutoffs per size-dependent type. Small n, so treat as a hint, not truth."""
    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    best_part = dict(PART_FRACTION_THRESHOLDS)
    for t in PART_FRACTION_THRESHOLDS:
        sub = [f for f in feats if f["type"] == t]
        if len(sub) < 5:
            continue
        cands = [(lo, hi) for lo, hi in itertools.combinations(grid, 2)]
        best_part[t] = list(max(cands, key=lambda c: score(sub, {**best_part, t: list(c)})[0]))
    return best_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="grid-search part-fraction thresholds")
    args = parser.parse_args()

    feats = collect_features()
    print(f"\nPart matched for {sum(f['part'] is not None for f in feats)}/{len(feats)} damages "
          f"(rest fall back to image fraction)")
    print("Matched parts:", dict(Counter(f["part"] for f in feats if f["part"])))

    _, _, preds = score(feats)
    report(feats, preds, "Current rules")

    if args.search:
        best = grid_search(feats)
        _, _, preds = score(feats, best, IMAGE_FRACTION_THRESHOLDS)
        report(feats, preds, "Grid-searched part thresholds (optimistic: tuned on the same labels)")
        print("\nSuggested PART_FRACTION_THRESHOLDS =", best)


if __name__ == "__main__":
    main()
