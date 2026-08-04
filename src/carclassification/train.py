"""
train.py
Trains a ResNet-50 classifier on the Stanford dataset and saves best model by validation accuracy.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

from src.carclassification.data_loader import get_dataloaders

def build_model(num_classes = 196):
    """
    Loads pretrained Resnet-50 model and swaps final layer for a new linear layer with num_classes outputs.

    Args:
        num_classes (int): Number of output classes for the new linear layer.
    Returns:
        model (torch.nn.Module): Modified ResNet-50 model with new final layer.
    """

    model = models.resnet50(weights='IMAGENET1K_V2')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(num_epochs=20, batch_size=32, learning_rate=1e-4, data_root="data/carclassificationdataset1", checkpoint_path="models/classifier/resnet50_stanford_cars_best.pth"):
    """
    Trains a ResNet-50 classifier on the Stanford Cars dataset and saves the best model based on validation accuracy.

    Args:
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for the DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        data_root (str): Root directory of the dataset.
        checkpoint_path (str): Path to save the best model checkpoint.

    Returns:
        str: Path to the saved best model checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=data_root, batch_size=batch_size
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (Val Acc: {val_acc:.4f})")

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best model saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train()