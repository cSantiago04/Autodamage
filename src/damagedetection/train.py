"""
train.py
Trains a Mask R-CNN damage detector on CarDD dataset and saves the best checkpoint to disk.
"""

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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, category_names = get_dataloaders(
        data_root=data_root, batch_size=batch_size
    )
    num_classes = len(category_names) + 1  # +1 for background

    model = build_model(num_classes).to(device)

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
    train()