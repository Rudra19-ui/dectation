#!/usr/bin/env python3
"""
Breast Cancer Detection Model Prediction System
Includes model saving, loading, and prediction functions
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Configuration
MODEL_PATH = "models/breast_cancer_detector_resnet50.pt"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# Data transforms (same as training)
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def create_model():
    """Create the model architecture"""
    print("🔧 Creating model architecture...")

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

    model = model.to(DEVICE)
    print("✅ Model architecture created")
    return model


def save_model(model, save_path=None):
    """Save the trained model"""
    if save_path is None:
        save_path = MODEL_PATH

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"💾 Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved successfully!")

    return save_path


def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        model_path = MODEL_PATH

    print(f"🔧 Loading model from {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or check the model path.")
        return None

    try:
        # Create model architecture
        model = create_model()

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully")

        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    try:
        # Load and convert image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        input_tensor = test_transform(image).unsqueeze(0)

        return input_tensor, image
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None, None


def predict_image(model, image_path):
    """Predict class for a single image"""
    print(f"🔍 Predicting for image: {image_path}")

    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    if input_tensor is None:
        return None

    # Make prediction
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    # Get results
    predicted_class = CLASS_NAMES[prediction.item()]
    confidence_score = confidence.item()
    all_probabilities = probabilities.cpu().numpy()[0]

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "all_probabilities": all_probabilities,
        "original_image": original_image,
    }


def display_prediction_result(result, image_path):
    """Display prediction results"""
    if result is None:
        print("❌ Prediction failed")
        return

    print("\n" + "=" * 60)
    print("🔍 PREDICTION RESULTS")
    print("=" * 60)
    print(f"📁 Image: {os.path.basename(image_path)}")
    print(f"🏷️  Predicted Class: {result['predicted_class']}")
    print(
        f"🎯 Confidence Score: {result['confidence_score']:.4f} ({result['confidence_score']*100:.2f}%)"
    )

    print(f"\n📊 Class Probabilities:")
    for i, (class_name, prob) in enumerate(
        zip(CLASS_NAMES, result["all_probabilities"])
    ):
        print(f"   {class_name}: {prob:.4f} ({prob*100:.2f}%)")

    # Determine confidence level
    confidence_level = (
        "High"
        if result["confidence_score"] > 0.8
        else "Medium" if result["confidence_score"] > 0.6 else "Low"
    )
    print(f"\n📈 Confidence Level: {confidence_level}")

    # Show image with prediction
    if result["original_image"]:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(result["original_image"])
        plt.title(f"Original Image\n{os.path.basename(image_path)}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        classes = CLASS_NAMES
        probs = result["all_probabilities"]
        colors = [
            (
                "green"
                if i == CLASS_NAMES.index(result["predicted_class"])
                else "lightgray"
            )
            for i in range(len(classes))
        ]

        bars = plt.bar(classes, probs, color=colors)
        plt.title(
            f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence_score']:.2%}"
        )
        plt.ylabel("Probability")
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_filename = f"prediction_result_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Prediction visualization saved as: {plot_filename}")
        plt.show()


def predict_batch(model, image_paths):
    """Predict for multiple images"""
    print(f"🔍 Predicting for {len(image_paths)} images...")

    results = []
    for i, image_path in enumerate(image_paths):
        print(
            f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}"
        )
        result = predict_image(model, image_path)
        if result:
            results.append({"image_path": image_path, "prediction": result})

    return results


def display_batch_results(results):
    """Display results for multiple images"""
    print("\n" + "=" * 80)
    print("📊 BATCH PREDICTION RESULTS")
    print("=" * 80)

    print(f"{'Image':<30} {'Predicted':<10} {'Confidence':<12} {'Status':<8}")
    print("-" * 80)

    for result in results:
        image_name = os.path.basename(result["image_path"])
        pred_class = result["prediction"]["predicted_class"]
        confidence = result["prediction"]["confidence_score"]

        # Truncate long image names
        if len(image_name) > 28:
            image_name = image_name[:25] + "..."

        print(
            f"{image_name:<30} {pred_class:<10} {confidence:.4f}      {'✅' if confidence > 0.7 else '⚠️'}"
        )


def main():
    """Main function for testing the prediction system"""
    print("🏥 Breast Cancer Detection - Prediction System")
    print("=" * 60)

    # Load model
    model = load_model()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    print(f"\n✅ Model loaded successfully on {DEVICE}")
    print(f"📊 Model can predict: {', '.join(CLASS_NAMES)}")

    # Example usage
    print("\n📋 Usage Examples:")
    print("1. Single image prediction:")
    print("   result = predict_image(model, 'path/to/image.jpg')")
    print("   display_prediction_result(result, 'path/to/image.jpg')")
    print("\n2. Batch prediction:")
    print("   results = predict_batch(model, ['img1.jpg', 'img2.jpg'])")
    print("   display_batch_results(results)")

    print("\n🎉 Prediction system ready!")


if __name__ == "__main__":
    main()
