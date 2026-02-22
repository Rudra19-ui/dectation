#!/usr/bin/env python3
"""
Generate Current Confusion Matrix for Breast Cancer Classification
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
MODEL_PATH = "breast_cancer_model.h5"
TEST_DATA_PATH = "dataset"  # Path to test data directory
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["benign", "malignant", "normal"]


def load_model():
    """Load the trained breast cancer classification model"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Available models in current directory:")
        for file in os.listdir("."):
            if file.endswith((".h5", ".keras")):
                print(f"  - {file}")
        return None

    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"📊 Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_test_data():
    """Load test data using ImageDataGenerator"""
    print("📂 Loading test data...")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Test data path not found: {TEST_DATA_PATH}")
        print("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  - {item}/")
        return None

    # Create data generator for test data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    try:
        # Load test data
        test_generator = test_datagen.flow_from_directory(
            TEST_DATA_PATH,
            target_size=INPUT_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
            workers=0,  # Disable multiprocessing for Windows compatibility
            use_multiprocessing=False,
        )

        print(f"✅ Test data loaded successfully!")
        print(f"📊 Test samples: {test_generator.samples}")
        print(f"📊 Classes found: {list(test_generator.class_indices.keys())}")

        return test_generator
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None


def get_predictions_and_labels(model, test_generator):
    """Get predictions and true labels from the model"""
    print("🔍 Generating predictions...")

    # Reset generator
    test_generator.reset()

    # Get predictions
    predictions = model.predict(
        test_generator, workers=0, use_multiprocessing=False, verbose=1
    )

    # Get true labels
    y_true = test_generator.classes

    # Convert predictions to class labels
    y_pred = np.argmax(predictions, axis=1)

    print(f"✅ Predictions generated for {len(y_true)} samples")
    print(f"📊 Prediction distribution: {np.bincount(y_pred)}")
    print(f"📊 True label distribution: {np.bincount(y_true)}")

    return y_true, y_pred, predictions


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
        "Current Confusion Matrix - Breast Cancer Classification",
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
    filename = f"confusion_matrix_current_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {filename}")

    plt.show()

    return cm


def display_detailed_metrics(y_true, y_pred, cm):
    """Display detailed classification metrics"""
    print("\n" + "=" * 80)
    print("📊 DETAILED CLASSIFICATION METRICS")
    print("=" * 80)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
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


def save_results(y_true, y_pred, cm):
    """Save results to CSV files"""
    print("\n💾 Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_filename = f"confusion_matrix_data_{timestamp}.csv"
    cm_df.to_csv(cm_filename)
    print(f"📊 Confusion matrix data saved as: {cm_filename}")

    # Save predictions vs actual
    results_df = pd.DataFrame(
        {
            "True_Label": [CLASS_NAMES[i] for i in y_true],
            "Predicted_Label": [CLASS_NAMES[i] for i in y_pred],
            "Correct": y_true == y_pred,
        }
    )

    results_filename = f"predictions_vs_actual_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"📊 Predictions vs actual saved as: {results_filename}")

    # Calculate and save summary statistics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    summary = {
        "timestamp": timestamp,
        "overall_accuracy": accuracy,
        "total_samples": len(y_true),
        "class_distribution": {
            CLASS_NAMES[i]: int(np.sum(y_true == i)) for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    import json

    summary_filename = f"model_evaluation_summary_{timestamp}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary statistics saved as: {summary_filename}")


def main():
    """Main function"""
    print("=" * 80)
    print("🔍 CURRENT CONFUSION MATRIX GENERATOR")
    print("🏥 Breast Cancer Classification Model Evaluation")
    print("=" * 80)
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Test Data: {TEST_DATA_PATH}")
    print(f"📊 Classes: {CLASS_NAMES}")
    print("=" * 80)

    # Load model
    model = load_model()
    if model is None:
        return

    # Load test data
    test_generator = load_test_data()
    if test_generator is None:
        return

    # Get predictions
    y_true, y_pred, predictions = get_predictions_and_labels(model, test_generator)

    # Create confusion matrix
    cm = create_confusion_matrix(y_true, y_pred)

    # Display detailed metrics
    display_detailed_metrics(y_true, y_pred, cm)

    # Save results
    save_results(y_true, y_pred, cm)

    print("\n" + "=" * 80)
    print("🎉 CONFUSION MATRIX GENERATION COMPLETE!")
    print("=" * 80)
    print("📊 Generated files:")
    print("  - Confusion matrix visualization (PNG)")
    print("  - Confusion matrix data (CSV)")
    print("  - Predictions vs actual (CSV)")
    print("  - Model evaluation summary (JSON)")
    print(
        "\n💡 Review the confusion matrix to understand your model's current performance!"
    )


if __name__ == "__main__":
    main()
