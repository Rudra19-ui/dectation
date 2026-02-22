import glob
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MammogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to PyTorch format (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(height=224, width=224),
                A.Normalize(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=224, width=224),
                A.Normalize(),
            ]
        )


def load_dataset(data_dir, split_csv):
    df = pd.read_csv(split_csv)
    image_paths = [os.path.join(data_dir, fname) for fname in df["filename"]]

    # Convert string labels to integers
    label_map = {"normal": 0, "benign": 1, "malignant": 2}
    labels = [label_map[label] for label in df["label"].values]

    return image_paths, labels


def get_dataloaders(data_dir, split_csvs, batch_size=16, num_workers=2):
    dataloaders = {}
    for phase in ["train", "val", "test"]:
        image_paths, labels = load_dataset(data_dir, split_csvs[phase])
        dataset = MammogramDataset(image_paths, labels, transform=get_transforms(phase))
        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
        )
    return dataloaders
