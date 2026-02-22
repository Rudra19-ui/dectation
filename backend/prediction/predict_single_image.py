#!/usr/bin/env python3
"""
Single Image Prediction Script for Breast Cancer Classification
Loads trained model and predicts class for a single test image
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
MODEL_PATH = "breast_cancer_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["benign", "malignant", "normal"]


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


def preprocess_image(image_path):
    """
    Preprocess a single image for prediction
    - Load and resize to 224x224
    - Convert to RGB if needed
    - Normalize pixel values to [0, 1]
    """
    try:
        # Load image
        img = load_img(image_path, target_size=INPUT_SIZE, color_mode="rgb")

        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, img
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None, None


def predict_image(model, image_path):
    """
    Predict class for a single image
    """
    print(f"🔍 Predicting for: {os.path.basename(image_path)}")

    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    if img_array is None:
        return None

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    # Get all class probabilities
    class_probabilities = predictions[0]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "original_image": original_img,
        "image_path": image_path,
    }


def display_prediction_result(result):
    """
    Display prediction results with visualization
    """
    if result is None:
        return

    print(f"\n📊 Prediction Results:")
    print(f"  Image: {os.path.basename(result['image_path'])}")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")

    print(f"\n📊 Class Probabilities:")
    for i, (class_name, prob) in enumerate(
        zip(CLASS_NAMES, result["class_probabilities"])
    ):
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    ax1.imshow(result["original_image"])
    ax1.set_title(f"Input Image\n{os.path.basename(result['image_path'])}")
    ax1.axis("off")

    # Class probabilities bar chart
    bars = ax2.bar(CLASS_NAMES, result["class_probabilities"])
    ax2.set_title("Class Probabilities")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)

    # Highlight predicted class
    predicted_idx = CLASS_NAMES.index(result["predicted_class"])
    bars[predicted_idx].set_color("red")

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, result["class_probabilities"])):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save plot
    plot_filename = (
        f"prediction_{os.path.splitext(os.path.basename(result['image_path']))[0]}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Prediction visualization saved as: {plot_filename}")
    plt.show()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Predict breast cancer class for a single image"
    )
    parser.add_argument("image_path", help="Path to the image file (PNG or JPG)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model")

    args = parser.parse_args()

    # Update model path if provided
    global MODEL_PATH
    MODEL_PATH = args.model

    print("=" * 60)
    print("🔍 Breast Cancer Classification - Single Image Prediction")
    print("=" * 60)

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return

    # Check file extension
    if not args.image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"❌ Unsupported file format. Please use PNG or JPG files.")
        return

    # Load model
    model = load_model()
    if model is None:
        return

    # Make prediction
    result = predict_image(model, args.image_path)

    # Display results
    display_prediction_result(result)

    print("\n" + "=" * 60)
    print("🎉 Prediction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
