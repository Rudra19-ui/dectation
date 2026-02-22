#!/usr/bin/env python3
"""
Breast Cancer Classification Prediction Script
Load trained model and predict on new images
"""

import os

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
    Load the trained model
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
    """
    try:
        # Load and resize image
        img = load_img(image_path, target_size=INPUT_SIZE)

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
    }


def display_prediction_result(result, image_path):
    """
    Display prediction results with visualization
    """
    if result is None:
        return

    print(f"\n📊 Prediction Results:")
    print(f"  Image: {os.path.basename(image_path)}")
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
    ax1.set_title(f"Input Image\n{os.path.basename(image_path)}")
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
        f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Prediction visualization saved as: {plot_filename}")
    plt.show()


def predict_batch(model, image_folder):
    """
    Predict for all images in a folder
    """
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return

    # Get all image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"❌ No image files found in {image_folder}")
        return

    print(f"🔍 Found {len(image_files)} images in {image_folder}")

    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        result = predict_image(model, image_path)
        if result:
            result["image_path"] = image_path
            results.append(result)

    return results


def display_batch_results(results):
    """
    Display summary of batch prediction results
    """
    if not results:
        return

    print(f"\n📊 Batch Prediction Summary:")
    print(f"  Total images processed: {len(results)}")

    # Count predictions by class
    class_counts = {}
    for result in results:
        predicted_class = result["predicted_class"]
        class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

    print(f"\n📊 Predictions by Class:")
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / len(results)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    # Average confidence
    avg_confidence = np.mean([r["confidence"] for r in results])
    print(f"\n📊 Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")


def main():
    """
    Main function
    """
    print("=" * 60)
    print("🔍 Breast Cancer Classification Prediction")
    print("=" * 60)

    # Load model
    model = load_model()
    if model is None:
        return

    # Example usage
    print("\n💡 Example usage:")
    print("1. Single image prediction:")
    print("   python predict_breast_cancer.py --image path/to/image.jpg")
    print("\n2. Batch prediction:")
    print("   python predict_breast_cancer.py --folder path/to/image/folder")
    print("\n3. Interactive mode:")
    print("   python predict_breast_cancer.py")

    # Check command line arguments
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--image" and len(sys.argv) > 2:
            image_path = sys.argv[2]
            if os.path.exists(image_path):
                result = predict_image(model, image_path)
                display_prediction_result(result, image_path)
            else:
                print(f"❌ Image not found: {image_path}")
        elif sys.argv[1] == "--folder" and len(sys.argv) > 2:
            folder_path = sys.argv[2]
            results = predict_batch(model, folder_path)
            display_batch_results(results)
        else:
            print("❌ Invalid arguments. Use --image or --folder")
    else:
        # Interactive mode
        print("\n🔍 Interactive Prediction Mode")
        print("Enter the path to an image file (or 'quit' to exit):")

        while True:
            try:
                user_input = input("\nImage path: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if os.path.exists(user_input):
                    result = predict_image(model, user_input)
                    display_prediction_result(result, user_input)
                else:
                    print(f"❌ File not found: {user_input}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
