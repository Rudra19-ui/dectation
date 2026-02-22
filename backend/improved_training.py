#!/usr/bin/env python3
"""
Improved Breast Cancer Classification Training
Addresses accuracy issues with better preprocessing and training
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")


class BreastCancerDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {"benign": 0, "malignant": 1, "normal": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["filename"]
        label = row["label"]

        # Load image
        image_path = self.images_dir / filename
        if not image_path.exists():
            # Try alternative paths
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_path = self.images_dir / f"{filename.split('.')[0]}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break

        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert label to index
        label_idx = self.class_to_idx[label]

        return image, label_idx


def create_transforms():
    """Create training and validation transforms"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_model(num_classes=3):
    """Create and configure the model"""
    # Use ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def train_model(model, train_loader, val_loader, num_epochs=20, device="cpu"):
    """Train the model with improved techniques"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "accuracy": val_accuracy,
                },
                "best_improved_model.pt",
            )
            print(f"  ✅ New best model saved! Accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_accuracy)

    return train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    # Classification report
    class_names = ["benign", "malignant", "normal"]
    report = classification_report(
        all_labels, all_predictions, target_names=class_names
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, report, cm


def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("improved_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("🏥 Improved Breast Cancer Classification Training")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists("data/train_enhanced.csv"):
        print(
            "❌ Training data not found. Please ensure data/train_enhanced.csv exists."
        )
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Create transforms
    train_transform, val_transform = create_transforms()

    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = BreastCancerDataset(
        "data/train_enhanced.csv", "data/images", train_transform
    )
    val_dataset = BreastCancerDataset(
        "data/val_enhanced.csv", "data/images", val_transform
    )
    test_dataset = BreastCancerDataset(
        "data/test_enhanced.csv", "data/images", val_transform
    )

    print(f"📈 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create model
    print("🤖 Creating model...")
    model = create_model(num_classes=3)
    model = model.to(device)

    # Train model
    print("🚀 Starting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=20, device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)

    # Load best model and evaluate
    print("📊 Loading best model for evaluation...")
    checkpoint = torch.load("best_improved_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    accuracy, report, cm = evaluate_model(model, test_loader, device)

    print(f"\n🎯 Final Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant", "normal"],
        yticklabels=["benign", "malignant", "normal"],
    )
    plt.title("Confusion Matrix - Improved Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("improved_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Training completed! Best model saved as 'best_improved_model.pt'")


if __name__ == "__main__":
    main()
