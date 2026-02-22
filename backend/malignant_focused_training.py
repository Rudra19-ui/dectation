#!/usr/bin/env python3
"""
Enhanced Breast Cancer Classification Training - Malignant Class Focused
Implements weighted loss functions and advanced techniques to improve malignant class detection
"""

import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "split_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
NUM_CLASSES = 3
CLASS_NAMES = ["benign", "malignant", "normal"]

# Training hyperparameters optimized for malignant detection
BATCH_SIZE = 24  # Slightly smaller batch for better gradient updates
EPOCHS = 40
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")


class FocalLoss(nn.Module):
    """
    Focal Loss implementation to handle class imbalance
    Focuses on hard-to-classify examples (especially malignant cases)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AdvancedAugmentationTransforms:
    """Advanced data augmentation specifically designed for medical images"""

    @staticmethod
    def get_malignant_focused_transforms():
        """Augmentation transforms that help with malignant detection"""
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224, padding=16),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
            ]
        )

    @staticmethod
    def get_validation_transforms():
        """Standard validation transforms"""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def calculate_class_weights_and_samples(dataset):
    """Calculate class weights and sample weights for balanced training"""
    # Count samples per class
    class_counts = {}
    for _, label in dataset:
        class_name = CLASS_NAMES[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"📊 Class distribution:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")

    # Calculate class weights (inverse frequency with emphasis on malignant)
    total_samples = sum(class_counts.values())
    class_weights = {}

    for i, class_name in enumerate(CLASS_NAMES):
        count = class_counts.get(class_name, 1)
        # Give extra weight to malignant class
        if class_name == "malignant":
            class_weights[i] = (
                total_samples / count
            ) * 1.5  # 1.5x multiplier for malignant
        else:
            class_weights[i] = total_samples / count

    print(f"📊 Class weights:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name}: {class_weights[i]:.4f}")

    # Create sample weights for WeightedRandomSampler
    sample_weights = []
    for _, label in dataset:
        sample_weights.append(class_weights[label])

    return class_weights, sample_weights


def create_enhanced_model(num_classes=3, dropout_rate=0.5):
    """Create enhanced ResNet50 model with better architecture for malignant detection"""
    # Use ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze early layers but allow more fine-tuning
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Enhanced classifier with attention mechanism
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        # First dense block
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        # Second dense block with residual connection
        nn.Dropout(dropout_rate * 0.7),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        # Final classification layer
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(512, num_classes),
    )

    return model


class MalignantFocusedTrainer:
    """Enhanced trainer specifically designed to improve malignant class detection"""

    def __init__(
        self, model, train_loader, val_loader, test_loader, class_weights, device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Convert class weights to tensor
        class_weights_tensor = torch.FloatTensor(list(class_weights.values())).to(
            device
        )

        # Initialize multiple loss functions
        self.criterions = {
            "weighted_ce": nn.CrossEntropyLoss(weight=class_weights_tensor),
            "focal": FocalLoss(alpha=class_weights_tensor, gamma=2.0),
        }

        # Optimizer with different learning rates for different parts
        self.optimizer = optim.AdamW(
            [
                {"params": model.layer3.parameters(), "lr": INITIAL_LR * 0.1},
                {"params": model.layer4.parameters(), "lr": INITIAL_LR * 0.5},
                {"params": model.fc.parameters(), "lr": INITIAL_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )

        # Advanced scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=INITIAL_LR,
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        # Tracking metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "malignant_precision": [],
            "malignant_recall": [],
            "malignant_f1": [],
        }

        self.best_malignant_f1 = 0.0
        self.best_overall_score = 0.0

    def calculate_malignant_metrics(self, y_true, y_pred):
        """Calculate specific metrics for malignant class"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None
        )
        return precision[0], recall[0], f1[0]

    def train_epoch(self, epoch):
        """Train for one epoch with multiple loss combination"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            # Combine multiple losses
            loss_weighted = self.criterions["weighted_ce"](outputs, labels)
            loss_focal = self.criterions["focal"](outputs, labels)

            # Combined loss with emphasis on focal loss for hard examples
            total_loss_batch = 0.6 * loss_weighted + 0.4 * loss_focal

            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += total_loss_batch.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {total_loss_batch.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate the model with detailed malignant metrics"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterions["weighted_ce"](outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        malignant_precision, malignant_recall, malignant_f1 = (
            self.calculate_malignant_metrics(all_labels, all_predictions)
        )

        return (
            total_loss / len(self.val_loader),
            accuracy,
            malignant_precision,
            malignant_recall,
            malignant_f1,
        )

    def train(self):
        """Main training loop"""
        print(f"🚀 Starting Enhanced Training for Malignant Detection")
        print("=" * 70)

        for epoch in range(EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_accuracy, mal_precision, mal_recall, mal_f1 = self.validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["malignant_precision"].append(mal_precision)
            self.history["malignant_recall"].append(mal_recall)
            self.history["malignant_f1"].append(mal_f1)

            print(f"\nEpoch {epoch+1}/{EPOCHS} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Malignant Precision: {mal_precision:.4f}")
            print(f"  Malignant Recall: {mal_recall:.4f}")
            print(f"  Malignant F1: {mal_f1:.4f}")

            # Save best model based on malignant performance
            # Combined score: 40% overall accuracy + 60% malignant F1
            combined_score = 0.4 * val_accuracy + 0.6 * mal_f1

            if combined_score > self.best_overall_score:
                self.best_overall_score = combined_score
                self.best_malignant_f1 = mal_f1

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "malignant_f1": mal_f1,
                        "combined_score": combined_score,
                        "class_weights": list(
                            self.criterions["weighted_ce"].weight.cpu().numpy()
                        ),
                    },
                    "best_malignant_focused_model.pt",
                )

                print(f"  🎯 NEW BEST MODEL! Combined Score: {combined_score:.4f}")

            print("-" * 70)

    def evaluate_detailed(self):
        """Comprehensive evaluation with detailed metrics"""
        print("🧪 Detailed Model Evaluation")
        print("=" * 50)

        # Load best model
        checkpoint = torch.load(
            "best_malignant_focused_model.pt", map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, all_predictions, target_names=CLASS_NAMES, digits=4
        )
        cm = confusion_matrix(all_labels, all_predictions)

        # Malignant-specific metrics
        mal_precision, mal_recall, mal_f1 = self.calculate_malignant_metrics(
            all_labels, all_predictions
        )

        print(f"📊 Final Test Results:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Malignant Precision: {mal_precision:.4f}")
        print(f"   Malignant Recall: {mal_recall:.4f}")
        print(f"   Malignant F1-Score: {mal_f1:.4f}")
        print(f"\n📋 Detailed Classification Report:")
        print(report)

        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": float(accuracy),
            "malignant_precision": float(mal_precision),
            "malignant_recall": float(mal_recall),
            "malignant_f1": float(mal_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                all_labels, all_predictions, target_names=CLASS_NAMES, output_dict=True
            ),
        }

        with open(
            f'malignant_focused_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            "w",
        ) as f:
            json.dump(results, f, indent=2)

        return accuracy, report, cm, all_predictions, all_labels

    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss curves
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(self.history["val_loss"], label="Val Loss", color="red")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Overall accuracy
        axes[0, 1].plot(
            self.history["val_accuracy"], label="Val Accuracy", color="green"
        )
        axes[0, 1].set_title("Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Malignant precision and recall
        axes[1, 0].plot(
            self.history["malignant_precision"], label="Precision", color="purple"
        )
        axes[1, 0].plot(
            self.history["malignant_recall"], label="Recall", color="orange"
        )
        axes[1, 0].set_title("Malignant Class: Precision & Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Malignant F1 score
        axes[1, 1].plot(
            self.history["malignant_f1"], label="Malignant F1", color="red", linewidth=2
        )
        axes[1, 1].set_title("Malignant Class: F1 Score")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            "malignant_focused_training_history.png", dpi=300, bbox_inches="tight"
        )
        plt.show()


def plot_enhanced_confusion_matrix(cm, class_names):
    """Plot enhanced confusion matrix with malignant focus"""
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Number of Samples"},
    )

    plt.title(
        "Enhanced Confusion Matrix - Malignant Focused Model",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    # Highlight malignant class errors
    total_malignant = cm[1].sum()
    malignant_errors = cm[1, 0] + cm[1, 2]  # Malignant classified as benign or normal

    plt.figtext(
        0.5,
        0.02,
        f"Malignant Misclassification Rate: {malignant_errors}/{total_malignant} ({malignant_errors/total_malignant*100:.1f}%)",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="red",
    )

    plt.tight_layout()
    plt.savefig(
        "enhanced_confusion_matrix_malignant_focused.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main training function"""
    print("🏥 Enhanced Breast Cancer Classification - Malignant Detection Focus")
    print("=" * 80)

    # Check data availability
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training data not found at {TRAIN_DIR}")
        print("Please ensure your dataset is properly organized in split_dataset/")
        return

    # Create enhanced transforms
    train_transform = AdvancedAugmentationTransforms.get_malignant_focused_transforms()
    val_transform = AdvancedAugmentationTransforms.get_validation_transforms()

    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(VAL_DIR, transform=val_transform)
    test_dataset = ImageFolder(TEST_DIR, transform=val_transform)

    print(f"📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Calculate class weights and create weighted sampler
    class_weights, sample_weights = calculate_class_weights_and_samples(train_dataset)

    # Create weighted sampler for balanced training
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Create enhanced model
    print("🤖 Creating enhanced model...")
    model = create_enhanced_model(num_classes=NUM_CLASSES, dropout_rate=0.5)

    # Initialize trainer
    trainer = MalignantFocusedTrainer(
        model, train_loader, val_loader, test_loader, class_weights, DEVICE
    )

    # Train model
    trainer.train()

    # Plot training history
    trainer.plot_training_history()

    # Comprehensive evaluation
    accuracy, report, cm, predictions, labels = trainer.evaluate_detailed()

    # Plot enhanced confusion matrix
    plot_enhanced_confusion_matrix(cm, CLASS_NAMES)

    print("\n✅ Enhanced training completed!")
    print(f"🎯 Best model saved as 'best_malignant_focused_model.pt'")
    print(f"📊 Best Malignant F1-Score: {trainer.best_malignant_f1:.4f}")


if __name__ == "__main__":
    main()
