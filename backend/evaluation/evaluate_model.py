#!/usr/bin/env python3
"""
Model Evaluation Script for Breast Cancer Detection
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Configuration
DATA_DIR = "split_dataset"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/breast_cancer_detector_resnet50.pt"
NUM_CLASSES = 3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# Data transforms
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model():
    """Load the trained model"""
    print(f"🔧 Loading model from {MODEL_PATH}")

    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        return None

    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    print("\n🧪 Evaluating model on test set...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_confidences = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Processed {batch_idx + 1}/{len(test_loader)} batches", end="\r"
                )

    print(f"\n✅ Evaluation complete!")
    return all_predictions, all_labels, all_probabilities, all_confidences


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    print("\n📊 Calculating metrics...")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def display_metrics(metrics):
    """Display comprehensive metrics"""
    print("\n" + "=" * 60)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")

    print(f"\n📋 Per-Class Performance:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name}:")
        print(f"     Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"     Recall:    {metrics['recall_per_class'][i]:.4f}")
        print(f"     F1 Score:  {metrics['f1_per_class'][i]:.4f}")


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    print("\n📊 Creating confusion matrix...")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    save_path = f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Confusion matrix saved as: {save_path}")
    plt.show()


def show_sample_predictions(
    all_predictions, all_labels, all_probabilities, all_confidences, num_samples=10
):
    """Show sample predictions with confidence values"""
    print(f"\n🔍 Sample Predictions (showing {num_samples} examples):")
    print("-" * 80)
    print(f"{'True':<10} {'Predicted':<10} {'Confidence':<12} {'Probabilities':<30}")
    print("-" * 80)

    indices = np.random.choice(
        len(all_predictions), min(num_samples, len(all_predictions)), replace=False
    )

    correct_count = 0
    for idx in indices:
        true_label = CLASS_NAMES[all_labels[idx]]
        pred_label = CLASS_NAMES[all_predictions[idx]]
        confidence = all_confidences[idx]
        probabilities = all_probabilities[idx]

        prob_str = " ".join(
            [f"{CLASS_NAMES[i]}:{prob:.3f}" for i, prob in enumerate(probabilities)]
        )

        is_correct = all_labels[idx] == all_predictions[idx]
        if is_correct:
            correct_count += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{true_label:<10} {pred_label:<10} {confidence:.4f}      {prob_str}")

    print("-" * 80)
    print(
        f"Sample Accuracy: {correct_count}/{num_samples} ({correct_count/num_samples*100:.1f}%)"
    )


def main():
    """Main evaluation function"""
    print("🏥 Breast Cancer Detection Model Evaluation")
    print("=" * 60)

    # Load model
    model = load_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Load test data
    print(f"📊 Loading test data from {TEST_DIR}")
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # Evaluate model
    all_predictions, all_labels, all_probabilities, all_confidences = evaluate_model(
        model, test_loader
    )

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions)

    # Display results
    display_metrics(metrics)

    # Plot confusion matrix
    plot_confusion_matrix(metrics["confusion_matrix"])

    # Show sample predictions
    show_sample_predictions(
        all_predictions, all_labels, all_probabilities, all_confidences
    )

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
