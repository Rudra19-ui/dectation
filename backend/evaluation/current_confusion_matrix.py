#!/usr/bin/env python3
"""
Generate Current Confusion Matrix for Breast Cancer Classification
Works with your trained PyTorch model and CSV-based dataset structure
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Configuration
MODEL_PATH = "best_improved_model.pt"  # Your best model
TEST_CSV = "data/test_enhanced.csv"  # Test labels
IMAGES_DIR = "data/images"  # Images directory
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


class BreastCancerDataset(Dataset):
    """Custom dataset class for breast cancer images"""

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
            # Try alternative extensions
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

        return image, label_idx, filename


def load_model():
    """Load the trained PyTorch model"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Available model files:")
        for file in os.listdir("."):
            if file.endswith(".pt"):
                print(f"  - {file}")
        return None

    try:
        # Create model architecture matching your improved training
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze early layers (as done in training)
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last few layers
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Replace the final layer with custom classifier
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

        # Load trained weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "accuracy" in checkpoint:
                    print(
                        f"📊 Model accuracy from checkpoint: {checkpoint['accuracy']:.2f}%"
                    )
                if "epoch" in checkpoint:
                    print(f"📊 Best epoch: {checkpoint['epoch']}")
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"📊 Model architecture: ResNet50 with enhanced classifier")
        print(f"📊 Classifier: 2048 → 512 → {NUM_CLASSES}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_test_data():
    """Load test data using custom dataset"""
    print("📂 Loading test data...")

    if not os.path.exists(TEST_CSV):
        print(f"❌ Test CSV not found: {TEST_CSV}")
        return None

    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return None

    # Data preprocessing transforms (matching training)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        # Load test dataset
        test_dataset = BreastCancerDataset(TEST_CSV, IMAGES_DIR, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # For Windows compatibility
        )

        print(f"✅ Test data loaded successfully!")
        print(f"📊 Test samples: {len(test_dataset)}")

        # Show class distribution
        test_df = pd.read_csv(TEST_CSV)
        class_distribution = test_df["label"].value_counts()
        print(f"📊 Test class distribution:")
        for class_name, count in class_distribution.items():
            print(f"   {class_name}: {count} samples")

        return test_loader, test_dataset
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None


def get_predictions_and_labels(model, test_loader, test_dataset):
    """Get predictions and true labels from the model"""
    print("🔍 Generating predictions...")

    y_true = []
    y_pred = []
    y_pred_proba = []
    filenames = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, batch_filenames) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probabilities.cpu().numpy())
            filenames.extend(batch_filenames)

            # Progress update
            if (batch_idx + 1) % 5 == 0:
                processed = min((batch_idx + 1) * BATCH_SIZE, len(test_dataset))
                print(f"   Processed {processed}/{len(test_dataset)} samples")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    print(f"✅ Predictions generated for {len(y_true)} samples")
    print(f"📊 Prediction distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(y_pred == i)
        print(f"   {class_name}: {count} predictions")

    return y_true, y_pred, y_pred_proba, filenames


def create_confusion_matrix(y_true, y_pred):
    """Create and display confusion matrix"""
    print("📈 Creating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with professional styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Number of Samples"},
        square=True,
        ax=ax1,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax1.set_title(
        "Current Confusion Matrix - Raw Counts", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual Class", fontsize=12, fontweight="bold")

    # Normalized confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Recall (True Positive Rate)"},
        square=True,
        ax=ax2,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax2.set_title(
        "Current Confusion Matrix - Normalized (Recall)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual Class", fontsize=12, fontweight="bold")

    # Calculate and display overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Add overall accuracy and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"Breast Cancer Classification - Current Performance\n"
        f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"Generated: {timestamp}",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    plt.tight_layout()

    # Save the confusion matrix
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"current_confusion_matrix_{timestamp_file}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {filename}")

    plt.show()

    return cm, cm_normalized


def display_detailed_metrics(y_true, y_pred, cm):
    """Display detailed classification metrics"""
    print("\n" + "=" * 80)
    print("📊 CURRENT MODEL PERFORMANCE METRICS")
    print("=" * 80)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class metrics
    print(f"\n📊 Per-Class Performance:")
    print("-" * 70)

    total_samples = len(y_true)

    for i, class_name in enumerate(CLASS_NAMES):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Support (actual samples of this class)
        support = tp + fn

        print(f"\n{class_name.upper()} CLASS:")
        print(
            f"  Support:         {support} samples ({support/total_samples*100:.1f}%)"
        )
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  Precision:       {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:          {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
        print(f"  Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")

    # Classification report
    print(f"\n📋 Scikit-learn Classification Report:")
    print("-" * 70)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save confusion matrix data
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{name}" for name in CLASS_NAMES],
        columns=[f"Pred_{name}" for name in CLASS_NAMES],
    )
    cm_filename = f"confusion_matrix_data_{timestamp}.csv"
    cm_df.to_csv(cm_filename)

    # Save detailed predictions
    results_df = pd.DataFrame(
        {
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "predicted_label": [CLASS_NAMES[i] for i in y_pred],
            "correct": y_true == y_pred,
        }
    )
    results_filename = f"detailed_predictions_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)

    # Save evaluation summary
    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "test_samples": len(y_true),
        "overall_accuracy": float(accuracy),
        "per_class_metrics": {},
    }

    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        evaluation_summary["per_class_metrics"][class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(tp + fn),
        }

    import json

    summary_filename = f"evaluation_summary_{timestamp}.json"
    with open(summary_filename, "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\n📁 Results saved:")
    print(f"   - Confusion matrix data: {cm_filename}")
    print(f"   - Detailed predictions: {results_filename}")
    print(f"   - Evaluation summary: {summary_filename}")


def main():
    """Main function to generate current confusion matrix"""
    print("=" * 80)
    print("🏥 BREAST CANCER CLASSIFICATION - CURRENT CONFUSION MATRIX")
    print("=" * 80)
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    print("🤖 Loading trained model...")
    model = load_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Load test data
    print("\n📂 Loading test data...")
    test_loader, test_dataset = load_test_data()
    if test_loader is None:
        print("❌ Failed to load test data. Exiting.")
        return

    # Generate predictions
    print("\n🔍 Generating predictions...")
    y_true, y_pred, y_pred_proba, filenames = get_predictions_and_labels(
        model, test_loader, test_dataset
    )

    # Create confusion matrix
    print("\n📊 Creating confusion matrix...")
    cm, cm_normalized = create_confusion_matrix(y_true, y_pred)

    # Display detailed metrics
    display_detailed_metrics(y_true, y_pred, cm)

    print(f"\n🎉 Confusion matrix generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
