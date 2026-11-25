"""
data_loader.py
Parses COCO-format damage annotations (CARDD dataset) and builds PyTorch DataLoaders.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

def load_coco_annotations(json_path):
    """
    Load COCO-format annotations from a JSON file.

    Args:
        json_path (str): Path to the COCO annotations JSON file.

    Returns:
        dict: A dictionary containing the COCO annotations.
    """

    with open(json_path) as f:
        return json.load(f)

def polygon_to_mask(segmentation, height, width):
    """
    Convert polygon segmentation to a binary mask.

    Args:
        segmentation (list): List of polygons (each polygon is a list of x, y coordinates).
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        np.ndarray: A binary mask of shape (height, width) where pixels inside the polygon are 1 and outside are 0.
    """

    mask = Image.new('L', (width, height), 0)
    
    for polygon in segmentation:
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)

def get_transform(train):
    """
    Get the transformation pipeline for the dataset.

    Args:
        train (bool): Whether the transformations are for training or validation/testing.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    transforms = [T.ToImage()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort(p=0.5))  # brightness/contrast/saturation jitter
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)

class DamageDataset(Dataset):
    """
    PyTorch Dataset for the CARDD dataset with COCO-format annotations. 
    Returns (image, target) pairs, where target is a dictionary containing 'boxes', 'labels', and 'masks', 'image_idx', 'area', 'iscrowd'.
    """

    def __init__(self, coco_data, img_dir, transform=None, include_empty=False):
        """
        Args:
            coco_data (dict): COCO-format annotations loaded from a JSON file.
            img_dir (str): Directory containing the images.
            transform (callable, optional): A function/transform to apply to the images and targets.
            include_empty (bool): Keep images with no annotations (negative examples).
        """
        self.img_dir = img_dir
        self.transform = transform

        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        self.annotations_by_image = {}
        for anno in coco_data['annotations']:
            self.annotations_by_image.setdefault(anno['image_id'], []).append(anno)

        self.image_ids = [img_id for img_id in self.images
                          if include_empty or img_id in self.annotations_by_image]

    def __len__(self):
        """
        returns the number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        When given an index, returns the corresponding image and its target annotations.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, target) where image is a tensor and target is a dictionary containing 'boxes', 'labels', 'masks', 'image_id', 'area', and 'iscrowd'.
        """
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        annos = self.annotations_by_image.get(image_id, [])

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for anno in annos:
            x, y, w, h = anno['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(anno['category_id'])
            masks.append(polygon_to_mask(anno['segmentation'], height, width))
            areas.append(anno['area'])
            iscrowd.append(anno.get('iscrowd', 0))

        if not annos:  # negative example: empty targets with the right shapes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = np.zeros((0, height, width), dtype=np.uint8)

        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': tv_tensors.Mask(torch.as_tensor(np.array(masks), dtype=torch.uint8)),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transform:
            img, target = self.transform(img, target)
        else:
            img = F.to_tensor(img)

        return img, target

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.

    Args:
        batch (list): A list of tuples where each tuple is (image, target).

    Returns:
        tuple: A tuple containing a list of images and a list of targets.
    """

    return tuple(zip(*batch))

def get_dataloaders(data_root="data/CarDD_release/CarDD_release/CarDD_COCO", batch_size=4,
                    img_dir_pattern="{split}2017", include_empty=False):
    """
    Builds train/val/test DataLoaders for any COCO-format segmentation dataset laid out as
    <data_root>/annotations/instances_<split>2017.json (CarDD's layout). Used for both
    damage detection (CarDD) and car-part segmentation (Carparts-seg).

    Args:
        data_root (str): Root directory of the dataset.
        batch_size (int): Number of samples in each batch.
        img_dir_pattern (str): Image folder for each split, relative to data_root.
            "{split}2017" for CarDD, "images/{split}" for Carparts-seg.
        include_empty (bool): Keep images without annotations (negatives) in train/val/test.

    Returns:
        tuple: (train_loader, val_loader, test_loader, category_names) where each loader is a DataLoader and category_names is a dictionary mapping category IDs to names.
    """
    annotations_dir = os.path.join(data_root, "annotations")

    train_coco = load_coco_annotations(os.path.join(annotations_dir, "instances_train2017.json"))
    val_coco = load_coco_annotations(os.path.join(annotations_dir, "instances_val2017.json"))
    test_coco = load_coco_annotations(os.path.join(annotations_dir, "instances_test2017.json"))

    img_dir = lambda split: os.path.join(data_root, img_dir_pattern.format(split=split))
    train_dataset = DamageDataset(train_coco, img_dir("train"), transform=get_transform(train=True),
                                  include_empty=include_empty)
    val_dataset = DamageDataset(val_coco, img_dir("val"), transform=get_transform(train=False),
                                include_empty=include_empty)
    test_dataset = DamageDataset(test_coco, img_dir("test"), transform=get_transform(train=False),
                                 include_empty=include_empty)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    category_names = {cat['id']: cat['name'] for cat in train_coco['categories']}

    return train_loader, val_loader, test_loader, category_names