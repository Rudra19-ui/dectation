#!/usr/bin/env python3
"""
Train Thermal/Ultrasound Breast Cancer Classification Model
"""

import json
import warnings
from datetime import datetime
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")


class ThermalBreastCancerDataset(Dataset):
    """Dataset for thermal/ultrasound breast cancer images"""

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.class_names = ["benign", "malignant", "normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert label to index
        label_idx = self.class_to_idx[label]

        return image, label_idx


def create_thermal_transforms():
    """Create transforms for thermal/ultrasound images"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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


def create_thermal_model():
    """Create model for thermal/ultrasound classification"""
    # Use EfficientNet-B0 for thermal images
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
        nn.Linear(256, 3),
    )

    return model


def train_thermal_model(model, train_loader, val_loader, device, num_epochs=30):
    """Train the thermal/ultrasound model"""
    print("🚀 Starting Thermal/Ultrasound Model Training")
    print("=" * 60)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
    }

    best_val_accuracy = 0.0

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

            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # Calculate balanced accuracy
        from sklearn.metrics import balanced_accuracy_score

        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_balanced_accuracy"].append(balanced_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
        print(f"  Val Balanced Accuracy: {balanced_acc:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_balanced_accuracy": balanced_acc,
                },
                "thermal_best_model.pt",
            )
            print(f"  ✅ New best model saved! (Accuracy: {val_accuracy:.2f}%)")

        print("-" * 40)

    return history


def evaluate_thermal_model(model, test_loader, device):
    """Evaluate the trained thermal model"""
    print("📊 Evaluating Thermal/Ultrasound Model")
    print("=" * 50)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    from sklearn.metrics import balanced_accuracy_score

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # Classification report
    class_names = ["benign", "malignant", "normal"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Thermal/Ultrasound Model Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("thermal_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def plot_thermal_training_history(history):
    """Plot training history for thermal model"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Training and validation loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Validation accuracy
    ax2.plot(history["val_accuracy"], label="Validation Accuracy", color="green")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Balanced accuracy
    ax3.plot(
        history["val_balanced_accuracy"], label="Balanced Accuracy", color="orange"
    )
    ax3.set_title("Balanced Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Balanced Accuracy (%)")
    ax3.legend()
    ax3.grid(True)

    # Combined metrics
    ax4.plot(history["val_accuracy"], label="Accuracy", color="green")
    ax4.plot(
        history["val_balanced_accuracy"], label="Balanced Accuracy", color="orange"
    )
    ax4.set_title("Accuracy Comparison")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("thermal_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main training function"""
    print("🔬 Thermal/Ultrasound Breast Cancer Classification Training")
    print("=" * 70)

    # Check if dataset is processed
    if not Path("data/thermal_train.csv").exists():
        print(
            "❌ Thermal dataset not processed. Please run process_thermal_dataset.py first."
        )
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Create transforms
    train_transform, val_transform = create_thermal_transforms()

    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ThermalBreastCancerDataset(
        "data/thermal_train.csv", transform=train_transform
    )
    val_dataset = ThermalBreastCancerDataset(
        "data/thermal_val.csv", transform=val_transform
    )
    test_dataset = ThermalBreastCancerDataset(
        "data/thermal_test.csv", transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Create model
    print("🏗️ Creating model...")
    model = create_thermal_model()
    model.to(device)

    # Train model
    print("🚀 Starting training...")
    history = train_thermal_model(
        model, train_loader, val_loader, device, num_epochs=30
    )

    # Load best model for evaluation
    print("📊 Loading best model for evaluation...")
    checkpoint = torch.load("thermal_best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate model
    results = evaluate_thermal_model(model, test_loader, device)

    # Plot training history
    plot_thermal_training_history(history)

    # Save training results
    training_results = {
        "model_type": "thermal_ultrasound",
        "best_val_accuracy": checkpoint["val_accuracy"],
        "best_val_balanced_accuracy": checkpoint["val_balanced_accuracy"],
        "test_accuracy": results["accuracy"],
        "test_balanced_accuracy": results["balanced_accuracy"],
        "training_history": history,
        "timestamp": datetime.now().isoformat(),
    }

    with open("thermal_training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📄 Results saved to: thermal_training_results.json")
    print(f"📊 Best validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"📊 Test accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
