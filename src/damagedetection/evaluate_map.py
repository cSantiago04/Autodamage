"""
evaluate_map.py
Standard COCO evaluation (mAP) of the damage detector on the CarDD test set, for
both boxes and masks, overall and per damage class. Loss alone doesn't say how
good the detections are; mAP is the metric the CarDD paper reports, so these
numbers are directly comparable to its baselines.

Also reports precision/recall at the confidence threshold the app actually uses
(default 0.5), which is what shows false positives and missed damage.

Usage (from repo root):
    python -m src.damagedetection.evaluate_map
    python -m src.damagedetection.evaluate_map --checkpoint models/damage_detector/other.pth
    python -m src.damagedetection.evaluate_map --data-root data/carpartsseg --img-dir-pattern "images/{split}"         --checkpoint models/part_detector/maskrcnn_carparts_best.pth          # part segmenter
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.damagedetection.prediction import class_names_for
from src.damagedetection.train import build_model

DATA_ROOT = "data/CarDD_release/CarDD_release/CarDD_COCO"


def run_inference(model, coco, img_dir, device, label_to_cat=None):
    """
    Returns COCO-format detection results (bbox + segm) for every test image.
    label_to_cat maps model label ids to ground-truth category ids (None = identity);
    labels mapped to None are dropped.
    """
    bbox_results, segm_results = [], []
    model.eval()
    with torch.no_grad():
        for img_id in tqdm(coco.getImgIds(), desc="Inference"):
            info = coco.loadImgs(img_id)[0]
            img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
            pred = model([F.to_tensor(img).to(device)])[0]

            masks = (pred["masks"][:, 0] > 0.5).cpu().numpy().astype(np.uint8)
            for box, label, score, m in zip(pred["boxes"].cpu(), pred["labels"].cpu(),
                                            pred["scores"].cpu(), masks):
                x1, y1, x2, y2 = box.tolist()
                cat = int(label) if label_to_cat is None else label_to_cat.get(int(label))
                if cat is None:
                    continue
                common = {"image_id": img_id, "category_id": cat, "score": float(score)}
                bbox_results.append({**common, "bbox": [x1, y1, x2 - x1, y2 - y1]})
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                segm_results.append({**common, "segmentation": rle})
    return bbox_results, segm_results


def coco_eval(coco_gt, results, iou_type):
    """Runs COCOeval; returns overall stats and per-class AP / AP50."""
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    print(f"\n----- {iou_type.upper()} -----")
    ev.summarize()

    # precision array: [IoU thresholds, recall points, classes, area ranges, max dets]
    precision = ev.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        name = coco_gt.loadCats(int(cat_id))[0]["name"]  # pycocotools rejects numpy ints
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_class[name] = {
            "AP": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan"),
            "AP50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else float("nan"),
        }
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "per_class": per_class}


def precision_recall_at_threshold(coco_gt, bbox_results, score_thresh=0.5, iou_thresh=0.5):
    """Per-class precision/recall when only detections >= score_thresh are kept (the app's view)."""
    from torchvision.ops import box_iou

    stats = {c["id"]: {"name": c["name"], "tp": 0, "fp": 0, "fn": 0} for c in coco_gt.loadCats(coco_gt.getCatIds())}
    dets_by_img = {}
    for d in bbox_results:
        if d["score"] >= score_thresh:
            dets_by_img.setdefault(d["image_id"], []).append(d)

    to_xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    for img_id in coco_gt.getImgIds():
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dets = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])
        for cat_id in stats:
            g = [to_xyxy(a["bbox"]) for a in gts if a["category_id"] == cat_id]
            p = [to_xyxy(d["bbox"]) for d in dets if d["category_id"] == cat_id]
            if not g:
                stats[cat_id]["fp"] += len(p)
                continue
            if not p:
                stats[cat_id]["fn"] += len(g)
                continue
            ious = box_iou(torch.tensor(p), torch.tensor(g))
            matched = set()
            for i in range(len(p)):  # greedy, highest score first
                j = int(ious[i].argmax())
                if ious[i, j] >= iou_thresh and j not in matched:
                    matched.add(j)
                    stats[cat_id]["tp"] += 1
                else:
                    stats[cat_id]["fp"] += 1
            stats[cat_id]["fn"] += len(g) - len(matched)

    out = {}
    for s in stats.values():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        out[s["name"]] = {"precision": tp / (tp + fp) if tp + fp else float("nan"),
                          "recall": tp / (tp + fn) if tp + fn else float("nan"),
                          "tp": tp, "fp": fp, "fn": fn}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/damage_detector/maskrcnn_cardd_best.pth")
    parser.add_argument("--split", default="test",
                        help="annotation file instances_<split>2017.json, e.g. val, test, test_hitl")
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--score-thresh", type=float, default=0.5, help="app's confidence cutoff for P/R")
    parser.add_argument("--rerun", action="store_true", help="ignore cached detections")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--img-dir-pattern", default="{split}2017")
    args = parser.parse_args()

    ann_path = os.path.join(args.data_root, "annotations", f"instances_{args.split}2017.json")
    img_dir = os.path.join(args.data_root, args.img_dir_pattern.format(split=args.split))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(ann_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Map the model's classes onto the ground truth's by name, so a model trained with a
    # different label set (e.g. the original CarDD model on the unified dataset) can be scored
    from src.damagedetection.build_damage_dataset import TO_UNIFIED
    model_classes = class_names_for(args.checkpoint)
    gt_ids = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    label_to_cat = {}
    for i, n in enumerate(model_classes, start=1):
        label_to_cat[i] = gt_ids.get(n, gt_ids.get(TO_UNIFIED.get(n)))
    model = build_model(num_classes=len(model_classes) + 1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.roi_heads.nms_thresh = args.nms_thresh
    model.to(device)

    os.makedirs("results", exist_ok=True)
    cache = f"results/damage_dets_{args.split}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json"
    if os.path.exists(cache) and not args.rerun:
        print(f"Using cached detections from {cache} (pass --rerun to recompute)")
        with open(cache) as f:
            bbox_results, segm_results = json.load(f)
    else:
        bbox_results, segm_results = run_inference(model, coco_gt, img_dir, device, label_to_cat)
        with open(cache, "w") as f:
            json.dump([bbox_results, segm_results], f)

    n_imgs = len(coco_gt.getImgIds())
    flagged = {d["image_id"] for d in bbox_results if d["score"] >= args.score_thresh}
    n_dets = sum(d["score"] >= args.score_thresh for d in bbox_results)
    print(f"\nImages with any damage >= {args.score_thresh}: {len(flagged)}/{n_imgs} "
          f"({n_dets} detections, {n_dets / max(n_imgs, 1):.2f} per image)")
    if not coco_gt.getAnnIds():  # negatives only: nothing to compute AP against
        print("(no ground-truth damage in this split: every detection above is a false positive)")
        return

    box = coco_eval(coco_gt, bbox_results, "bbox")
    segm = coco_eval(coco_gt, segm_results, "segm")
    pr = precision_recall_at_threshold(coco_gt, bbox_results, args.score_thresh)

    print(f"\n{'class':<14} {'box AP':>7} {'box AP50':>9} {'mask AP':>8} {'mask AP50':>10} "
          f"{'P@' + str(args.score_thresh):>7} {'R@' + str(args.score_thresh):>7}   tp / fp / fn")
    for name in box["per_class"]:
        b, s, p = box["per_class"][name], segm["per_class"][name], pr[name]
        print(f"{name:<14} {b['AP']:>7.3f} {b['AP50']:>9.3f} {s['AP']:>8.3f} {s['AP50']:>10.3f} "
              f"{p['precision']:>7.2f} {p['recall']:>7.2f}   {p['tp']} / {p['fp']} / {p['fn']}")
    print(f"{'ALL':<14} {box['AP']:>7.3f} {box['AP50']:>9.3f} {segm['AP']:>8.3f} {segm['AP50']:>10.3f}")

    os.makedirs("results", exist_ok=True)
    out = f"results/damage_map_{args.split}_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json"
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split, "bbox": box, "segm": segm,
                   "precision_recall": pr, "score_thresh": args.score_thresh}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
