"""
evaluate_part_matching.py
The question that matters for repair estimates: is each damage assigned to the right part?

Uses the 120 held-out Humans in the Loop test photos, which have BOTH part and damage
labels. For every labeled damage:
    true part      = assign_part(damage mask, ground-truth part masks)
    predicted part = assign_part(damage mask, part masks predicted by the model)
and reports how often they agree (ground-truth damage masks are used, so this isolates
the part model + matching, not the damage detector).

Usage (from repo root):
    python -m src.damagedetection.evaluate_part_matching --checkpoint models/part_detector/maskrcnn_hitl_finetuned_best.pth
    python -m src.damagedetection.evaluate_part_matching --checkpoint models/part_detector/maskrcnn_carparts_best.pth
"""

import argparse
import contextlib
import io
import os
from collections import Counter, defaultdict

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from src.damagedetection import part_prediction
from src.damagedetection.repair_action import assign_part

ROOT = "data/Car parts and car damages dataset"
IMG_DIR = os.path.join(ROOT, "Car damages dataset/File1/img")

# Carparts-seg model names -> HITL names, so the older model can be scored on the same scale
CARPARTS_TO_HITL = {"front_light": "headlight", "back_light": "taillight", "front_glass": "windshield"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        parts_gt = COCO(os.path.join(ROOT, "annotations", "instances_test2017.json"))
        dmg_gt = COCO(os.path.join(ROOT, "annotations", "damage_instances_test2017.json"))
    part_prediction.load_model(args.checkpoint)

    by_name = {i["file_name"]: i["id"] for i in parts_gt.dataset["images"]}
    stats = Counter()
    per_part = defaultdict(Counter)
    confusions = Counter()

    for dimg in tqdm(dmg_gt.dataset["images"], desc="Matching"):
        pid = by_name[dimg["file_name"]]
        gt_parts = [{"part": parts_gt.loadCats(a["category_id"])[0]["name"], "mask": parts_gt.annToMask(a).astype(bool)}
                    for a in parts_gt.loadAnns(parts_gt.getAnnIds(imgIds=pid))]
        pred = part_prediction.predict_parts(Image.open(os.path.join(IMG_DIR, dimg["file_name"])))
        for p in pred:
            p["part"] = CARPARTS_TO_HITL.get(p["part"], p["part"])

        for a in dmg_gt.loadAnns(dmg_gt.getAnnIds(imgIds=dimg["id"])):
            m = dmg_gt.annToMask(a).astype(bool)
            true_part = assign_part(m, gt_parts)["part"]
            if true_part is None:
                continue  # damage not on any labeled part (e.g. on the license plate)
            got = assign_part(m, pred)["part"]
            stats["total"] += 1
            per_part[true_part]["total"] += 1
            if got is None:
                stats["unmatched"] += 1
            elif got == true_part:
                stats["correct"] += 1
                per_part[true_part]["correct"] += 1
            else:
                stats["wrong"] += 1
                confusions[(true_part, got)] += 1

    n = stats["total"]
    print(f"\n{args.checkpoint}")
    print(f"Damages on a labeled part: {n}")
    print(f"  correct part : {stats['correct'] / n:.1%}")
    print(f"  wrong part   : {stats['wrong'] / n:.1%}")
    print(f"  no part found: {stats['unmatched'] / n:.1%}")
    print("\nPer true part (correct / total):")
    for part, c in sorted(per_part.items(), key=lambda x: -x[1]["total"]):
        print(f"  {part:<14} {c['correct']:>4} / {c['total']:<4} {c['correct'] / c['total']:.0%}")
    print("\nMost common mix-ups (true -> predicted):")
    for (t, g), k in confusions.most_common(8):
        print(f"  {t:<14} -> {g:<14} {k}")


if __name__ == "__main__":
    main()
