"""
train.py
Trains a Mask R-CNN on a COCO-format segmentation dataset and saves the best checkpoint.
Used for the damage detector (CarDD, default) and the car-part segmenter (Carparts-seg).

Usage (from repo root):
    python -m src.damagedetection.train                       # CarDD damage detector
    python -m src.damagedetection.train --data-root data/carpartsseg --img-dir-pattern "images/{split}"         --checkpoint models/part_detector/maskrcnn_carparts_best.pth
"""

import argparse
import json

import os 
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.damagedetection.data_loader import get_dataloaders

def build_model(num_classes):
    """
    Loads pretrained Mask R-CNN and swaps box/mask heads.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device, print_every=100):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % print_every == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)} | Loss: {losses.item():.4f}")

    epoch_time = time.time() - epoch_start
    print(f"  Epoch time: {epoch_time/60:.1f} min")

    return total_loss / len(data_loader)

def evaluate_loss(model, data_loader, device):
    """
    Returns average loss on dataset.
    """
    model.train() 
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(data_loader)

def train(
    #Changed to 20 epochs
    num_epochs=20,
    batch_size=4,
    learning_rate=0.005,
    lr_step_size=7,
    lr_gamma=0.1,
    data_root="data/CarDD_release/CarDD_release/CarDD_COCO",
    checkpoint_path="models/damage_detector/maskrcnn_cardd_best.pth",
    img_dir_pattern="{split}2017",
    max_train_batches=None,
    init_from=None,
    include_empty=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, category_names = get_dataloaders(
        data_root=data_root, batch_size=batch_size, img_dir_pattern=img_dir_pattern,
        include_empty=include_empty
    )
    if max_train_batches:  # quick smoke test on a small subset
        from torch.utils.data import DataLoader, Subset
        n = max_train_batches * batch_size
        train_loader = DataLoader(Subset(train_loader.dataset, range(n)), batch_size=batch_size,
                                  shuffle=True, collate_fn=train_loader.collate_fn)
        val_loader = DataLoader(Subset(val_loader.dataset, range(n)), batch_size=batch_size,
                                shuffle=False, collate_fn=val_loader.collate_fn)
    num_classes = len(category_names) + 1  # +1 for background

    model = build_model(num_classes)
    if init_from:
        # Start from another checkpoint (e.g. the Carparts-seg part model). Everything that
        # matches is loaded; the class-specific heads differ in size and stay freshly initialized.
        state = torch.load(init_from, map_location="cpu")
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        print(f"Initialized from {init_from}: {len(compatible)}/{len(own)} tensors loaded")
    model = model.to(device)

    # Save class names next to the checkpoint so prediction code knows what each id means
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(os.path.splitext(checkpoint_path)[0] + ".classes.json", "w") as f:
        json.dump([category_names[k] for k in sorted(category_names)], f)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (Val Loss: {val_loss:.4f})")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr}")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/CarDD_release/CarDD_release/CarDD_COCO")
    parser.add_argument("--img-dir-pattern", default="{split}2017")
    parser.add_argument("--checkpoint", default="models/damage_detector/maskrcnn_cardd_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-step-size", type=int, default=7)
    parser.add_argument("--max-train-batches", type=int, default=None, help="smoke test on a subset")
    parser.add_argument("--init-from", default=None, help="checkpoint to start from (heads that don't fit are reset)")
    parser.add_argument("--include-empty", action="store_true", help="train on images without annotations (negatives)")
    a = parser.parse_args()
    train(num_epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr, lr_step_size=a.lr_step_size,
          data_root=a.data_root, checkpoint_path=a.checkpoint, img_dir_pattern=a.img_dir_pattern,
          max_train_batches=a.max_train_batches, init_from=a.init_from,
          include_empty=a.include_empty)