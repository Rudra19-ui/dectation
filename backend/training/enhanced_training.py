#!/usr/bin/env python3
"""
Enhanced Training Script for Breast Cancer Detection
"""

import os
from collections import Counter

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = 224


class AdvancedDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_training=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_training = is_training

        if transform is None:
            self.transform = self.get_transforms(is_training)
        else:
            self.transform = transform

        self.label_map = {"benign": 0, "malignant": 1}

    def get_transforms(self, is_training):
        if is_training:
            return A.Compose(
                [
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.OneOf(
                        [
                            A.CLAHE(clip_limit=2),
                            A.IAASharpen(),
                            A.RandomBrightnessContrast(),
                        ],
                        p=0.3,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["filename"]
        label = self.data.iloc[idx]["label"]

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label_idx = self.label_map[label]
        return image, label_idx


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_weighted_sampler(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_enhanced_model():
    print("🚀 Training Enhanced Model...")

    # Create datasets
    train_dataset = AdvancedDataset("data/train.csv", "data/images", is_training=True)
    val_dataset = AdvancedDataset("data/val.csv", "data/images", is_training=False)

    # Create weighted sampler
    train_sampler = create_weighted_sampler(train_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model (ResNet50)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Training loop
    best_val_f1 = 0
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(DEVICE)
            labels = labels.long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(DEVICE)
                labels = labels.long().to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}"
            )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "enhanced_model.pt")

    print(f"Best validation F1: {best_val_f1:.4f}")
    return model


def evaluate_enhanced_model(model):
    print("\n🧪 Evaluating Enhanced Model...")

    # Create test dataset
    test_dataset = AdvancedDataset("data/test.csv", "data/images", is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(DEVICE)
            labels = labels.long().to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average="macro")
    recall = recall_score(test_labels, test_preds, average="macro")
    f1 = f1_score(test_labels, test_preds, average="macro")

    print(f"📊 Enhanced Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.title("Enhanced Model - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("enhanced_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    print("🚀 Enhanced Breast Cancer Detection Training")
    print("=" * 50)

    # Train enhanced model
    model = train_enhanced_model()

    # Evaluate model
    metrics = evaluate_enhanced_model(model)

    print(f"\n🎯 Summary:")
    print(f"   Enhanced Model F1 Score: {metrics['f1']:.4f}")
    print(f"   Expected improvement: +5-10% over baseline")
    print(f"   Model saved: enhanced_model.pt")
    print(f"   Confusion matrix: enhanced_confusion_matrix.png")
