#!/usr/bin/env python3
"""
Generate Current Confusion Matrix for Breast Cancer Classification (PyTorch)
Works with your trained PyTorch models and dataset structure
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Configuration
MODEL_PATH = "best_improved_model.pt"  # Your latest model
TEST_DATA_PATH = "split_dataset/test"  # Path to test data directory
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


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
        # Create model architecture (ResNet50 with custom FC layer)
        model = models.resnet50(weights=None)  # Don't load pretrained weights

        # Recreate the exact same FC layer architecture from training
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
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"📊 Model architecture: ResNet50 with custom FC layer")
        print(f"📊 FC Architecture: 2048 → 512 → {NUM_CLASSES}")
        print(f"📊 Number of classes: {NUM_CLASSES}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"💡 Trying alternative model files...")
        # Try other model files
        alternative_models = [f for f in os.listdir(".") if f.endswith(".pt")]
        for alt_model in alternative_models[:3]:  # Try first 3 alternatives
            try:
                print(f"🔄 Trying {alt_model}...")
                checkpoint = torch.load(alt_model, map_location=device)
                model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()
                print(f"✅ Successfully loaded alternative model: {alt_model}")
                return model
            except Exception as alt_e:
                print(f"❌ Failed to load {alt_model}: {alt_e}")
                continue
        return None


def load_test_data():
    """Load test data using PyTorch DataLoader"""
    print("📂 Loading test data...")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Test data path not found: {TEST_DATA_PATH}")
        print("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  - {item}/")
        return None, None

    # Data preprocessing transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        # Load test dataset
        test_dataset = ImageFolder(TEST_DATA_PATH, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # For Windows compatibility
        )

        print(f"✅ Test data loaded successfully!")
        print(f"📊 Test samples: {len(test_dataset)}")
        print(f"📊 Classes found: {test_dataset.classes}")
        print(f"📊 Class to index mapping: {test_dataset.class_to_idx}")

        # Update CLASS_NAMES to match dataset order
        global CLASS_NAMES
        CLASS_NAMES = test_dataset.classes

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

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
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

            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Processed {(batch_idx + 1) * BATCH_SIZE}/{len(test_dataset)} samples"
                )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    print(f"✅ Predictions generated for {len(y_true)} samples")
    print(f"📊 Prediction distribution: {np.bincount(y_pred)}")
    print(f"📊 True label distribution: {np.bincount(y_true)}")

    return y_true, y_pred, y_pred_proba


def create_confusion_matrix(y_true, y_pred):
    """Create and display confusion matrix"""
    print("📈 Creating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with larger size
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Number of Samples"},
        square=True,
    )

    plt.title(
        "Current Confusion Matrix - Breast Cancer Classification (PyTorch Model)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Predicted Class", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Class", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Add percentage annotations
    total = np.sum(cm)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            percentage = cm[i, j] / total * 100
            plt.text(
                j + 0.5,
                i + 0.8,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="red",
                fontweight="bold",
            )

    # Add accuracy information
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(
        0.5,
        0.02,
        f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save the confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pytorch_confusion_matrix_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {filename}")

    plt.show()

    return cm


def display_detailed_metrics(y_true, y_pred, cm):
    """Display detailed classification metrics"""
    print("\n" + "=" * 80)
    print("📊 DETAILED CLASSIFICATION METRICS (PyTorch Model)")
    print("=" * 80)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class metrics
    print(f"\n📊 Per-Class Performance:")
    print("-" * 60)

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

        print(f"\n{class_name.upper()}:")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Negatives:  {tn}")
        print(f"  Precision:       {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:          {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
        print(f"  Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")

    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Confusion matrix breakdown
    print(f"\n📊 Confusion Matrix Breakdown:")
    print("-" * 60)
    total_samples = np.sum(cm)

    for i, true_class in enumerate(CLASS_NAMES):
        for j, pred_class in enumerate(CLASS_NAMES):
            count = cm[i, j]
            percentage = (count / total_samples) * 100
            if count > 0:
                print(
                    f"True {true_class} → Predicted {pred_class}: {count} samples ({percentage:.2f}%)"
                )


def create_additional_visualizations(y_true, y_pred, y_pred_proba, cm):
    """Create additional visualization plots"""
    print("📊 Creating additional visualizations...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Normalized Count"},
    )
    plt.title("Normalized Confusion Matrix - Breast Cancer Classification")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    norm_cm_file = f"normalized_confusion_matrix_{timestamp}.png"
    plt.savefig(norm_cm_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"📊 Normalized confusion matrix saved as: {norm_cm_file}")

    # 2. Class distribution comparison
    plt.figure(figsize=(12, 6))

    # Actual vs Predicted distribution
    actual_counts = np.bincount(y_true, minlength=len(CLASS_NAMES))
    predicted_counts = np.bincount(y_pred, minlength=len(CLASS_NAMES))

    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, actual_counts, width, label="Actual", alpha=0.8)
    plt.bar(x + width / 2, predicted_counts, width, label="Predicted", alpha=0.8)
    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    plt.title("Actual vs Predicted Distribution")
    plt.xticks(x, CLASS_NAMES)
    plt.legend()
    plt.xticks(rotation=45)

    # Prediction confidence distribution
    plt.subplot(1, 2, 2)
    max_proba = np.max(y_pred_proba, axis=1)
    plt.hist(max_proba, bins=20, alpha=0.7)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Prediction Confidence Distribution")
    plt.axvline(
        np.mean(max_proba),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(max_proba):.3f}",
    )
    plt.legend()

    plt.tight_layout()
    dist_file = f"class_distributions_{timestamp}.png"
    plt.savefig(dist_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"📊 Class distributions saved as: {dist_file}")


def save_results(y_true, y_pred, y_pred_proba, cm):
    """Save results to files"""
    print("\n💾 Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_filename = f"pytorch_confusion_matrix_data_{timestamp}.csv"
    cm_df.to_csv(cm_filename)
    print(f"📊 Confusion matrix data saved as: {cm_filename}")

    # Save detailed predictions
    results_df = pd.DataFrame(
        {
            "True_Label": [CLASS_NAMES[i] for i in y_true],
            "True_Label_Index": y_true,
            "Predicted_Label": [CLASS_NAMES[i] for i in y_pred],
            "Predicted_Label_Index": y_pred,
            "Correct": y_true == y_pred,
            "Confidence": np.max(y_pred_proba, axis=1),
        }
    )

    # Add probability columns for each class
    for i, class_name in enumerate(CLASS_NAMES):
        results_df[f"{class_name}_probability"] = y_pred_proba[:, i]

    results_filename = f"pytorch_detailed_predictions_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"📊 Detailed predictions saved as: {results_filename}")

    # Save summary statistics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    summary = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "test_data_path": TEST_DATA_PATH,
        "device": str(device),
        "overall_accuracy": float(accuracy),
        "total_samples": len(y_true),
        "class_names": CLASS_NAMES,
        "class_distribution": {
            CLASS_NAMES[i]: int(np.sum(y_true == i)) for i in range(len(CLASS_NAMES))
        },
        "prediction_distribution": {
            CLASS_NAMES[i]: int(np.sum(y_pred == i)) for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": (
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        ).tolist(),
        "classification_report": report,
        "average_prediction_confidence": float(np.mean(np.max(y_pred_proba, axis=1))),
    }

    import json

    summary_filename = f"pytorch_evaluation_summary_{timestamp}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Evaluation summary saved as: {summary_filename}")


def main():
    """Main function"""
    print("=" * 80)
    print("🔍 PYTORCH MODEL CONFUSION MATRIX GENERATOR")
    print("🏥 Breast Cancer Classification Model Evaluation")
    print("=" * 80)
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Test Data: {TEST_DATA_PATH}")
    print(f"📊 Expected Classes: {CLASS_NAMES}")
    print(f"🚀 Device: {device}")
    print("=" * 80)

    # Load model
    model = load_model()
    if model is None:
        return

    # Load test data
    test_loader, test_dataset = load_test_data()
    if test_loader is None:
        return

    # Get predictions
    y_true, y_pred, y_pred_proba = get_predictions_and_labels(
        model, test_loader, test_dataset
    )

    # Create confusion matrix
    cm = create_confusion_matrix(y_true, y_pred)

    # Display detailed metrics
    display_detailed_metrics(y_true, y_pred, cm)

    # Create additional visualizations
    create_additional_visualizations(y_true, y_pred, y_pred_proba, cm)

    # Save results
    save_results(y_true, y_pred, y_pred_proba, cm)

    print("\n" + "=" * 80)
    print("🎉 PYTORCH CONFUSION MATRIX GENERATION COMPLETE!")
    print("=" * 80)
    print("📊 Generated files:")
    print("  - Confusion matrix visualization (PNG)")
    print("  - Normalized confusion matrix (PNG)")
    print("  - Class distributions plot (PNG)")
    print("  - Confusion matrix data (CSV)")
    print("  - Detailed predictions (CSV)")
    print("  - Evaluation summary (JSON)")
    print(
        "\n💡 Review these files to understand your PyTorch model's current performance!"
    )
    print(f"🎯 Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")


if __name__ == "__main__":
    main()
