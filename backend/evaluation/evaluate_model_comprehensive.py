#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Breast Cancer Classification
Generates classification report and confusion matrix using scikit-learn
Includes accuracy, precision, recall, and F1-score for each class
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
MODEL_PATH = "breast_cancer_model.h5"
TEST_DATA_PATH = "dataset"  # Path to test data directory
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASS_NAMES)


def load_model():
    """
    Load the trained breast cancer classification model
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please train the model first using train_breast_cancer_cnn.py")
        return None

    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_test_data():
    """
    Load test data using ImageDataGenerator
    """
    print("📂 Loading test data...")

    # Create data generator for test data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",  # Use validation split for testing
        shuffle=False,
        workers=0,  # Disable multiprocessing for Windows compatibility
        use_multiprocessing=False,
    )

    print(f"✅ Test data loaded successfully!")
    print(f"📊 Test samples: {test_generator.samples}")
    print(f"📊 Classes: {test_generator.class_indices}")

    return test_generator


def get_predictions_and_labels(model, test_generator):
    """
    Get predictions and true labels from the model
    """
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

    # Get prediction probabilities for each class
    y_pred_proba = predictions

    print(f"✅ Predictions generated for {len(y_true)} samples")

    return y_true, y_pred, y_pred_proba


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics
    """
    print("📊 Calculating metrics...")

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(CLASS_NAMES):
        per_class_metrics[class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": support[i],
        }

    # Macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    metrics = {
        "accuracy": accuracy,
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
        },
    }

    return metrics


def generate_classification_report(y_true, y_pred):
    """
    Generate detailed classification report
    """
    print("📋 Generating classification report...")

    # Generate scikit-learn classification report
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
    )

    return report


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix
    """
    print("📈 Plotting confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
    )

    plt.title(
        "Confusion Matrix - Breast Cancer Classification",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Add text annotations
    total = np.sum(cm)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            percentage = cm[i, j] / total * 100
            plt.text(
                j + 0.5,
                i + 0.7,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )

    plt.tight_layout()

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"confusion_matrix_{timestamp}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {save_path}")
    plt.show()

    return cm


def plot_roc_curves(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curves for each class
    """
    print("📈 Plotting ROC curves...")

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green"]
    for i, color in enumerate(colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        "ROC Curves - Breast Cancer Classification", fontsize=16, fontweight="bold"
    )
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"roc_curves_{timestamp}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 ROC curves saved as: {save_path}")
    plt.show()

    return roc_auc


def plot_precision_recall_curves(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curves for each class
    """
    print("📈 Plotting Precision-Recall curves...")

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

    # Calculate Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        pr_auc[i] = auc(recall[i], precision[i])

    # Plot Precision-Recall curves
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green"]
    for i, color in enumerate(colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"{CLASS_NAMES[i]} (AUC = {pr_auc[i]:.3f})",
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(
        "Precision-Recall Curves - Breast Cancer Classification",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"precision_recall_curves_{timestamp}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Precision-Recall curves saved as: {save_path}")
    plt.show()

    return pr_auc


def create_metrics_summary(metrics, classification_report_dict, roc_auc, pr_auc):
    """
    Create a comprehensive metrics summary
    """
    print("📋 Creating metrics summary...")

    # Create summary DataFrame
    summary_data = []

    for i, class_name in enumerate(CLASS_NAMES):
        summary_data.append(
            {
                "Class": class_name,
                "Precision": metrics["per_class"][class_name]["precision"],
                "Recall": metrics["per_class"][class_name]["recall"],
                "F1-Score": metrics["per_class"][class_name]["f1_score"],
                "Support": metrics["per_class"][class_name]["support"],
                "ROC AUC": roc_auc[i],
                "PR AUC": pr_auc[i],
            }
        )

    # Add macro and weighted averages
    summary_data.append(
        {
            "Class": "Macro Avg",
            "Precision": metrics["macro_avg"]["precision"],
            "Recall": metrics["macro_avg"]["recall"],
            "F1-Score": metrics["macro_avg"]["f1_score"],
            "Support": np.sum(
                [metrics["per_class"][c]["support"] for c in CLASS_NAMES]
            ),
            "ROC AUC": np.mean(list(roc_auc.values())),
            "PR AUC": np.mean(list(pr_auc.values())),
        }
    )

    summary_data.append(
        {
            "Class": "Weighted Avg",
            "Precision": metrics["weighted_avg"]["precision"],
            "Recall": metrics["weighted_avg"]["recall"],
            "F1-Score": metrics["weighted_avg"]["f1_score"],
            "Support": np.sum(
                [metrics["per_class"][c]["support"] for c in CLASS_NAMES]
            ),
            "ROC AUC": np.mean(list(roc_auc.values())),
            "PR AUC": np.mean(list(pr_auc.values())),
        }
    )

    summary_df = pd.DataFrame(summary_data)

    # Save summary to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"📊 Metrics summary saved as: {csv_path}")

    return summary_df


def print_detailed_results(metrics, classification_report_dict, summary_df):
    """
    Print detailed evaluation results
    """
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)

    # Overall accuracy
    print(
        f"\n🎯 Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
    )

    # Per-class results
    print(f"\n📋 Per-Class Results:")
    print("-" * 60)
    for class_name in CLASS_NAMES:
        class_metrics = metrics["per_class"][class_name]
        print(f"{class_name.upper():>10}:")
        print(
            f"  Precision: {class_metrics['precision']:.4f} ({class_metrics['precision']*100:.2f}%)"
        )
        print(
            f"  Recall:    {class_metrics['recall']:.4f} ({class_metrics['recall']*100:.2f}%)"
        )
        print(
            f"  F1-Score:  {class_metrics['f1_score']:.4f} ({class_metrics['f1_score']*100:.2f}%)"
        )
        print(f"  Support:   {class_metrics['support']}")
        print()

    # Macro and weighted averages
    print(f"📊 Macro Averages:")
    print(
        f"  Precision: {metrics['macro_avg']['precision']:.4f} ({metrics['macro_avg']['precision']*100:.2f}%)"
    )
    print(
        f"  Recall:    {metrics['macro_avg']['recall']:.4f} ({metrics['macro_avg']['recall']*100:.2f}%)"
    )
    print(
        f"  F1-Score:  {metrics['macro_avg']['f1_score']:.4f} ({metrics['macro_avg']['f1_score']*100:.2f}%)"
    )

    print(f"\n📊 Weighted Averages:")
    print(
        f"  Precision: {metrics['weighted_avg']['precision']:.4f} ({metrics['weighted_avg']['precision']*100:.2f}%)"
    )
    print(
        f"  Recall:    {metrics['weighted_avg']['recall']:.4f} ({metrics['weighted_avg']['recall']*100:.2f}%)"
    )
    print(
        f"  F1-Score:  {metrics['weighted_avg']['f1_score']:.4f} ({metrics['weighted_avg']['f1_score']*100:.2f}%)"
    )

    # Print summary table
    print(f"\n📋 Summary Table:")
    print("-" * 80)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Print scikit-learn classification report
    print(f"\n📋 Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def save_evaluation_results(
    metrics,
    classification_report_dict,
    confusion_matrix_result,
    roc_auc,
    pr_auc,
    summary_df,
):
    """
    Save all evaluation results to files
    """
    print("💾 Saving evaluation results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results to JSON
    results = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "test_data_path": TEST_DATA_PATH,
        "overall_accuracy": metrics["accuracy"],
        "per_class_metrics": metrics["per_class"],
        "macro_averages": metrics["macro_avg"],
        "weighted_averages": metrics["weighted_avg"],
        "confusion_matrix": confusion_matrix_result.tolist(),
        "roc_auc_scores": roc_auc,
        "pr_auc_scores": pr_auc,
        "classification_report": classification_report_dict,
    }

    json_path = f"evaluation_results_{timestamp}.json"
    import json

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Detailed results saved as: {json_path}")


def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of breast cancer classification model"
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model")
    parser.add_argument(
        "--test_data", default=TEST_DATA_PATH, help="Path to test data directory"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Update paths if provided
    global MODEL_PATH, TEST_DATA_PATH
    MODEL_PATH = args.model
    TEST_DATA_PATH = args.test_data

    print("=" * 80)
    print("🔍 COMPREHENSIVE MODEL EVALUATION - Breast Cancer Classification")
    print("=" * 80)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Test data path: {TEST_DATA_PATH}")
    print(f"📊 Classes: {CLASS_NAMES}")
    print("=" * 80)

    # Load model
    model = load_model()
    if model is None:
        return

    # Load test data
    test_generator = load_test_data()

    # Get predictions and labels
    y_true, y_pred, y_pred_proba = get_predictions_and_labels(model, test_generator)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    # Generate classification report
    classification_report_dict = generate_classification_report(y_true, y_pred)

    # Generate plots (if not skipped)
    if not args.no_plots:
        # Plot confusion matrix
        confusion_matrix_result = plot_confusion_matrix(y_true, y_pred)

        # Plot ROC curves
        roc_auc = plot_roc_curves(y_true, y_pred_proba)

        # Plot Precision-Recall curves
        pr_auc = plot_precision_recall_curves(y_true, y_pred_proba)
    else:
        # Calculate without plotting
        confusion_matrix_result = confusion_matrix(y_true, y_pred)

        # Calculate ROC AUC
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        roc_auc = {}
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr, tpr)

        # Calculate PR AUC
        pr_auc = {}
        for i in range(NUM_CLASSES):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            pr_auc[i] = auc(recall, precision)

    # Create metrics summary
    summary_df = create_metrics_summary(
        metrics, classification_report_dict, roc_auc, pr_auc
    )

    # Print detailed results
    print_detailed_results(metrics, classification_report_dict, summary_df)

    # Save all results
    save_evaluation_results(
        metrics,
        classification_report_dict,
        confusion_matrix_result,
        roc_auc,
        pr_auc,
        summary_df,
    )

    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)
    print("📊 Generated files:")
    print(f"  - Confusion matrix plot")
    print(f"  - ROC curves plot")
    print(f"  - Precision-Recall curves plot")
    print(f"  - Evaluation summary CSV")
    print(f"  - Detailed results JSON")
    print(
        "\n💡 Use these results to assess model performance and identify areas for improvement."
    )


if __name__ == "__main__":
    main()
