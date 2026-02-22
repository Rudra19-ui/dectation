#!/usr/bin/env python3
"""
Fix Model Accuracy - Address False Positive Issues
Improved training with better data handling and validation
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

warnings.filterwarnings("ignore")


class BalancedBreastCancerDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, balance_classes=True):
        self.data = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {"benign": 0, "malignant": 1, "normal": 2}

        # Balance classes if requested
        if balance_classes:
            self.balance_dataset()

    def balance_dataset(self):
        """Balance the dataset to prevent bias"""
        class_counts = self.data["label"].value_counts()
        min_count = class_counts.min()

        balanced_data = []
        for class_name in self.class_to_idx.keys():
            class_data = self.data[self.data["label"] == class_name]
            if len(class_data) > min_count:
                # Sample randomly to match minimum count
                class_data = class_data.sample(n=min_count, random_state=42)
            balanced_data.append(class_data)

        self.data = pd.concat(balanced_data, ignore_index=True)
        print(f"📊 Balanced dataset: {len(self.data)} samples")
        print(f"   Class distribution: {self.data['label'].value_counts().to_dict()}")

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


def create_improved_transforms():
    """Create more robust transforms"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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


def create_improved_model(num_classes=3):
    """Create a more robust model architecture"""
    # Use EfficientNet-B0 for better feature extraction
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers for fine-tuning
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Replace the final layer with a more robust classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    return model


def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalance"""
    class_counts = dataset.data["label"].value_counts()
    total_samples = len(dataset)
    class_weights = {
        class_name: total_samples / (len(class_counts) * count)
        for class_name, count in class_counts.items()
    }

    # Convert to tensor weights
    weights = [
        class_weights[dataset.data.iloc[i]["label"]] for i in range(len(dataset))
    ]
    return torch.FloatTensor(weights)


def train_improved_model(model, train_loader, val_loader, num_epochs=30, device="cpu"):
    """Train with improved techniques"""
    # Use weighted loss to handle class imbalance
    class_weights = torch.tensor([1.0, 2.0, 1.0]).to(
        device
    )  # Higher weight for malignant
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Use AdamW with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_balanced_accuracies = []

    best_balanced_accuracy = 0.0

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
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_balanced_accuracies.append(val_balanced_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2%}")
        print(f"  Val Balanced Accuracy: {val_balanced_accuracy:.2%}")

        # Save best model based on balanced accuracy
        if val_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = val_balanced_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "balanced_accuracy": val_balanced_accuracy,
                    "accuracy": val_accuracy,
                },
                "fixed_improved_model.pt",
            )
            print(
                f"  ✅ New best model saved! Balanced Accuracy: {val_balanced_accuracy:.2%}"
            )

        scheduler.step(val_balanced_accuracy)

    return train_losses, val_losses, val_accuracies, val_balanced_accuracies


def evaluate_improved_model(model, test_loader, device="cpu"):
    """Evaluate with detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    # Classification report
    class_names = ["benign", "malignant", "normal"]
    report = classification_report(
        all_labels, all_predictions, target_names=class_names
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, balanced_accuracy, report, cm, all_probabilities


def plot_improved_training_history(
    train_losses, val_losses, val_accuracies, val_balanced_accuracies
):
    """Plot improved training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Plot balanced accuracy
    ax3.plot(val_balanced_accuracies, label="Balanced Accuracy", color="orange")
    ax3.set_title("Balanced Accuracy (Better for Imbalanced Data)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Balanced Accuracy")
    ax3.legend()
    ax3.grid(True)

    # Plot both accuracies
    ax4.plot(val_accuracies, label="Accuracy")
    ax4.plot(val_balanced_accuracies, label="Balanced Accuracy")
    ax4.set_title("Accuracy Comparison")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("improved_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("🔧 Fixing Model Accuracy Issues")
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

    # Create improved transforms
    train_transform, val_transform = create_improved_transforms()

    # Create balanced datasets
    print("📊 Loading balanced datasets...")
    train_dataset = BalancedBreastCancerDataset(
        "data/train_enhanced.csv", "data/images", train_transform, balance_classes=True
    )
    val_dataset = BalancedBreastCancerDataset(
        "data/val_enhanced.csv", "data/images", val_transform, balance_classes=True
    )
    test_dataset = BalancedBreastCancerDataset(
        "data/test_enhanced.csv", "data/images", val_transform, balance_classes=True
    )

    print(f"📈 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create improved model
    print("🤖 Creating improved model...")
    model = create_improved_model(num_classes=3)
    model = model.to(device)

    # Train improved model
    print("🚀 Starting improved training...")
    train_losses, val_losses, val_accuracies, val_balanced_accuracies = (
        train_improved_model(
            model, train_loader, val_loader, num_epochs=30, device=device
        )
    )

    # Plot training history
    plot_improved_training_history(
        train_losses, val_losses, val_accuracies, val_balanced_accuracies
    )

    # Load best model and evaluate
    print("📊 Loading best model for evaluation...")
    checkpoint = torch.load("fixed_improved_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    accuracy, balanced_accuracy, report, cm, probabilities = evaluate_improved_model(
        model, test_loader, device
    )

    print(f"\n🎯 Final Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_accuracy:.4f}")
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
    plt.title("Confusion Matrix - Fixed Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("fixed_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Fixed training completed! Best model saved as 'fixed_improved_model.pt'")
    print(f"📊 Model Performance:")
    print(f"   - Accuracy: {accuracy:.2%}")
    print(f"   - Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"   - Epoch: {checkpoint['epoch']}")


if __name__ == "__main__":
    main()
