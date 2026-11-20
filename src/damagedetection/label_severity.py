"""
label_severity.py
Small labeling tool: shows CarDD test-set damages one at a time (ground-truth masks
highlighted) and records your severity judgment. These labels are what the
severity rules get measured and tuned against.

Keys:  1 = minor   2 = moderate   3 = severe   s = skip/unsure   b = back   q = quit
Labels are saved after every key press to labels/severity_labels.csv, so you can
quit and resume anytime.

Guide (think "what would a body shop do?"):
  minor    - cosmetic: small scratch or ding; buff out, touch up, or paintless dent repair
  moderate - needs body work and repainting of the panel, but the part is kept
  severe   - the part needs replacing (torn, crushed, shattered, broken) or it's structural

Usage (from repo root):
    python -m src.damagedetection.label_severity            # 100 damages, stratified by type
    python -m src.damagedetection.label_severity --n 150
"""

import argparse
import contextlib
import csv
import io
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

DATA_ROOT = "data/CarDD_release/CarDD_release/CarDD_COCO"
OUT_PATH = "labels/severity_labels.csv"
FIELDS = ["ann_id", "image_id", "file_name", "damage_type", "severity", "labeled_at"]
KEYS = {"1": "minor", "2": "moderate", "3": "severe", "s": "skip"}

# Share of the sample per damage type. Size-dependent types get most of the labels,
# since glass/lamp/tire severity is fixed by the rules.
MIX = {"dent": 0.30, "scratch": 0.30, "crack": 0.20, "lamp broken": 0.10,
       "glass shatter": 0.05, "tire flat": 0.05}


def sample_annotations(coco, n, seed=0):
    rng = random.Random(seed)
    cats = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    picked = []
    for name, share in MIX.items():
        ids = sorted(coco.getAnnIds(catIds=[cats[name]]))
        picked += rng.sample(ids, min(len(ids), round(n * share)))
    rng.shuffle(picked)
    return picked


def load_labels():
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH, newline="") as f:
        return {int(r["ann_id"]): r for r in csv.DictReader(f)}


def save_labels(labels):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in labels.values():
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(os.path.join(DATA_ROOT, "annotations", f"instances_{args.split}2017.json"))
    img_dir = os.path.join(DATA_ROOT, f"{args.split}2017")
    queue = sample_annotations(coco, args.n)
    labels = load_labels()
    state = {"i": next((i for i, a in enumerate(queue) if a not in labels), len(queue))}

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))

    def show():
        for ax in (ax_full, ax_zoom):
            ax.clear()
            ax.axis("off")
        if state["i"] >= len(queue):
            fig.suptitle(f"Done! {len(labels)} labels saved to {OUT_PATH}. Close the window.")
            fig.canvas.draw_idle()
            return
        ann = coco.loadAnns(queue[state["i"]])[0]
        info = coco.loadImgs(ann["image_id"])[0]
        img = np.asarray(Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB"))
        mask = coco.annToMask(ann).astype(bool)
        overlay = img.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

        ax_full.imshow(overlay)
        x, y, w, h = ann["bbox"]
        ax_full.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="yellow", lw=2))
        ax_full.set_title("Whole photo (damage in red, yellow box)")

        pad = 0.5 * max(w, h)
        x1, y1 = int(max(0, x - pad)), int(max(0, y - pad))
        x2, y2 = int(min(img.shape[1], x + w + pad)), int(min(img.shape[0], y + h + pad))
        ax_zoom.imshow(img[y1:y2, x1:x2])
        ax_zoom.set_title("Zoomed in (no overlay)")

        cat = coco.loadCats(ann["category_id"])[0]["name"]
        done = sum(1 for a in queue if a in labels)
        prev = labels.get(ann["id"], {}).get("severity")
        fig.suptitle(f"[{state['i'] + 1}/{len(queue)}, {done} labeled]   damage type: {cat.upper()}"
                     + (f"   (current label: {prev})" if prev else "")
                     + "\n1 = minor    2 = moderate    3 = severe    s = skip    b = back    q = quit",
                     fontsize=12)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
            return
        if event.key == "b":
            state["i"] = max(0, state["i"] - 1)
        elif event.key in KEYS and state["i"] < len(queue):
            ann = coco.loadAnns(queue[state["i"]])[0]
            labels[ann["id"]] = {
                "ann_id": ann["id"], "image_id": ann["image_id"],
                "file_name": coco.loadImgs(ann["image_id"])[0]["file_name"],
                "damage_type": coco.loadCats(ann["category_id"])[0]["name"],
                "severity": KEYS[event.key], "labeled_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_labels(labels)
            state["i"] += 1
        else:
            return
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    print(f"{sum(1 for a in queue if a in labels and labels[a]['severity'] != 'skip')} usable labels in {OUT_PATH}")


if __name__ == "__main__":
    main()
