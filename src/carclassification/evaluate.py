"""
evaluate.py
Loads a trained ResNet-50 classifier and evaluates it on the held-out test set of the Stanford Cars dataset, reporting accuracy and optional confusion matrix.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.carclassification.data_loader import get_dataloaders
from src.carclassification.train import build_model

def evaluate(checkpoint_path="models/classifier/resnet50_stanford_cars_best.pth", data_root="data/carclassificationdataset1", batch_size=32):
    """
    Evaluates a trained ResNet-50 classifier

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        data_root (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        test_loss (float): Average loss on the test set.
        test_acc (float): Accuracy on the test set.
        all_labels (list): True labels for the test set.
        all_preds (list): Predicted labels for the test set.
        class_names (list): List of class names corresponding to class indices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, class_names = get_dataloaders(
        data_root=data_root, batch_size=batch_size
    )

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_correct = 0
    test_total = 0
    test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    return test_loss, test_acc, all_labels, all_preds, class_names

def get_top_confusions(all_labels, all_preds, class_names, top_n=15):
    """"
    Computes the confusion matrix and returns the top N most confused class pairs

    Args:
        all_labels (list): True labels for the test set.
        all_preds (list): Predicted labels for the test set.
        class_names (list): List of class names corresponding to class indices.
        top_n (int): Number of top confused class pairs to return.

    Returns:
        df_confusions (pd.DataFrame): DataFrame containing the top confused class pairs.
    """

    num_classes = len(class_names)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)  # Zero out the diagonal to ignore correct predictions

    print(f"Total misclassifications: {cm.sum()}")

    confusions = []
    for true_idx, pred_idx in zip(*np.where(cm > 0)):
        confusions.append({
            "true Class": class_names[true_idx],
            "predicted Class": class_names[pred_idx],
            "count": cm[true_idx, pred_idx]
        })

    if not confusions:
        print("No confusions found")
        return pd.DataFrame(columns=['true_class', 'predicted_class', 'count'])

    df_confusions = pd.DataFrame(confusions).sort_values(by="count", ascending=False).head(top_n)
    return df_confusions

if __name__ == "__main__":
    test_loss, test_acc, all_labels, all_preds, class_names = evaluate()

    print("\nTop confused class pairs:")
    top_confusions = get_top_confusions(all_labels, all_preds, class_names)
    print(top_confusions.to_string(index=False))