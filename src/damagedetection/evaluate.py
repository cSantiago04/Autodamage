"""
evaluate.py
Loads trained damage detection checkpoint and evaluates it on the held-out CarDD Test set.
"""

import torch
import numpy as np
import pandas as pd
from torchvision.ops import box_iou

from src.damagedetection.data_loader import get_dataloaders
from src.damagedetection.train import build_model, evaluate_loss


def evaluate(
    checkpoint_path="models/damage_detector/maskrcnn_cardd_best.pth",
    data_root="data/CarDD_release/CarDD_release/CarDD_COCO",
    batch_size=4,
    nms_thresh=0.3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, category_names = get_dataloaders(data_root=data_root, batch_size=batch_size)
    num_classes = len(category_names) + 1

    model = build_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.roi_heads.nms_thresh = nms_thresh

    test_loss = evaluate_loss(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss, model, test_loader, category_names


def get_confusion_pairs(model, data_loader, device, category_names,
                          confidence_threshold=0.5, iou_threshold=0.5):
    """
    Matches predicted boxes to ground truth boxes by IoU overlap, then
    compares their labels. Unmatched predictions count as false positives
    ("background" -> predicted class); unmatched ground truth counts as
    missed detections (true class -> "background").
    """
    model.eval()
    pairs = []  # list of (true_label_name, pred_label_name)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                gt_boxes = target['boxes']
                gt_labels = target['labels']

                keep = prediction['scores'] > confidence_threshold
                pred_boxes = prediction['boxes'][keep].cpu()
                pred_labels = prediction['labels'][keep].cpu()

                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    continue

                matched_gt = set()
                matched_pred = set()

                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    ious = box_iou(gt_boxes, pred_boxes)  # [num_gt, num_pred]

                    # Greedily match highest-IoU pairs first
                    while True:
                        max_iou = ious.max().item() if ious.numel() > 0 else 0
                        if max_iou < iou_threshold:
                            break
                        gt_idx, pred_idx = np.unravel_index(ious.argmax(), ious.shape)
                        gt_idx, pred_idx = gt_idx.item(), pred_idx.item()

                        true_name = category_names[gt_labels[gt_idx].item()]
                        pred_name = category_names[pred_labels[pred_idx].item()]
                        pairs.append((true_name, pred_name))

                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)
                        ious[gt_idx, :] = -1  # remove this gt/pred from further matching
                        ious[:, pred_idx] = -1

                # Unmatched ground truth = missed detections
                for i in range(len(gt_boxes)):
                    if i not in matched_gt:
                        true_name = category_names[gt_labels[i].item()]
                        pairs.append((true_name, "MISSED (background)"))

                # Unmatched predictions = false positives
                for i in range(len(pred_boxes)):
                    if i not in matched_pred:
                        pred_name = category_names[pred_labels[i].item()]
                        pairs.append(("FALSE POSITIVE (background)", pred_name))

    return pairs


def summarize_confusions(pairs, top_n=15):
    """Returns the most common (true, predicted) pairs, excluding exact matches."""
    df = pd.DataFrame(pairs, columns=['true_class', 'predicted_class'])
    mistakes = df[df['true_class'] != df['predicted_class']]
    summary = mistakes.value_counts().reset_index(name='count')
    return summary.head(top_n)


if __name__ == "__main__":
    test_loss, model, test_loader, category_names = evaluate()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nMatching predictions to ground truth for confusion analysis...")
    pairs = get_confusion_pairs(model, test_loader, device, category_names)

    print("\nTop confusions (including missed detections and false positives):")
    top_confusions = summarize_confusions(pairs)
    print(top_confusions.to_string(index=False))