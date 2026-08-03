"""
data_loader.py
Loads data for the Stanford Cards dataset, and builds PyTorch DataLoaders for the classification stage.
"""

import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def parse_annotations(devkit_path, annos_filename, has_labels=True):
    """
    Parses Stanford Cars .mat annotation file into a Dataframe.

    Args:
        devkit_path (str): Path to the devkit directory containing the .mat file.
        annos_filename (str): Name of the .mat annotation file.
        has_labels (bool): Whether the dataset has labels (True for training, False for testing).

    Returns:
        Dataframe with columns: filename, x1, y1, x2, y2, class_id (if has_labels is True)
    """

    annos = loadmat(os.path.join(devkit_path, annos_filename))
    data = []

    for anno in annos['annotations'][0]:
        if has_labels:
            data.append({
                'filename': anno[5][0],
                'x1': anno[0][0][0],
                'y1': anno[1][0][0],
                'x2': anno[2][0][0],
                'y2': anno[3][0][0],
                'class_id': anno[4][0][0],
            })
        else:
            data.append({
                'filename': anno[4][0],
                'x1': anno[0][0][0],
                'y1': anno[1][0][0],
                'x2': anno[2][0][0],
                'y2': anno[3][0][0],
            })

    return pd.DataFrame(data)

def load_class_names(devkit_path):
    """
    Loads the class_names from the cars_meta.mat file.

    Args:
        devkit_path (str): Path to the devkit directory containing the cars_meta.mat file.

    Returns:
        List of class names.
    """
    meta = loadmat(os.path.join(devkit_path, 'cars_meta.mat'))
    class_names = [c[0] for c in meta['class_names'][0]]
    return class_names

class CarDataset(Dataset):
    """Custom Dataset for loading Stanford Cars images and labels."""

    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (DataFrame): DataFrame containing image filenames and bounding box coordinates.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets the image and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (image, label) pair.
        """
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert("RGB")
        img = img.crop((row['x1'], row['y1'], row['x2'], row['y2']))

        if self.transform:
            img = self.transform(img)

        label = row['class_id'] - 1  # shift to 0-indexed for PyTorch
        return img, label

def get_dataloaders(data_root = "data/carclassificationdataset1", batch_size=32, val_test_split=0.3):
    """
    Builds Dataloaders for the Stanford Cars dataset.

    Args:
        data_root (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoaders.
        val_test_split (float): Fraction of the training set to use for validation.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing.
    """

    # Define paths to the devkit and image directories
    devkit_path = os.path.join(data_root, "car_devkit", "devkit")
    img_dir = os.path.join(data_root, "cars_train", "cars_train")

    # Load class names and parse annotations
    class_names = load_class_names(devkit_path)
    df = parse_annotations(devkit_path, "cars_train_annos.mat", has_labels=True)
    df['class_name'] = df['class_id'].apply(lambda cid: class_names[cid - 1])

    train_df, temp_df = train_test_split(df, test_size=val_test_split, stratify=df['class_id'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class_id'], random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CarDataset(train_df, img_dir, transform=train_transform)
    val_dataset = CarDataset(val_df, img_dir, transform=eval_transform)
    test_dataset = CarDataset(test_df, img_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names

