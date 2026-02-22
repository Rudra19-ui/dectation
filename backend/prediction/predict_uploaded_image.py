#!/usr/bin/env python3
"""
Predict Breast Cancer from Uploaded Image
Command-line tool for predicting breast cancer from image files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


class BreastCancerPredictor:
    def __init__(self, model_path="quick_enhanced_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.class_names = ["benign", "malignant", "normal"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained PyTorch model"""
        try:
            # Create model architecture (ResNet50)
            self.model = models.resnet50(pretrained=False)
            num_classes = 3
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Check if it's a state dict or wrapped in a dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"🖥️ Using device: {self.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def predict_image(self, image_path):
        """Predict class for an image file"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "class": self.class_names[predicted_class],
                "confidence": confidence,
                "probabilities": probabilities[0].cpu().numpy(),
                "image_size": image.size,
            }

        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None


def display_prediction(image_path, prediction):
    """Display prediction results in a formatted way"""
    print("\n" + "=" * 60)
    print("🏥 BREAST CANCER CLASSIFICATION RESULTS")
    print("=" * 60)

    print(f"📸 Image: {image_path}")
    print(
        f"📏 Size: {prediction['image_size'][0]} x {prediction['image_size'][1]} pixels"
    )
    print()

    # Main prediction
    print(f"🎯 PREDICTION: {prediction['class'].upper()}")
    print(f"📊 Confidence: {prediction['confidence']:.2%}")
    print()

    # All probabilities
    print("📈 CLASS PROBABILITIES:")
    for i, (class_name, prob) in enumerate(
        zip(["benign", "malignant", "normal"], prediction["probabilities"])
    ):
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   {class_name:10} {bar} {prob:.2%}")

    print()

    # Medical advice
    if prediction["class"] == "malignant":
        print("⚠️  MEDICAL ADVICE:")
        print("   This image shows characteristics that may indicate malignancy.")
        print("   IMMEDIATE medical consultation is strongly recommended.")
        print("   Please consult with a qualified radiologist or oncologist.")
    elif prediction["class"] == "benign":
        print("⚠️  MEDICAL ADVICE:")
        print("   This image shows characteristics that may indicate benign findings.")
        print(
            "   Regular monitoring and follow-up with healthcare provider recommended."
        )
        print("   Consult with a qualified radiologist for proper assessment.")
    else:
        print("✅ MEDICAL ADVICE:")
        print("   No significant abnormalities detected in this image.")
        print("   Continue with regular screening as recommended by your doctor.")

    print()
    print("=" * 60)
    print("⚠️  DISCLAIMER: This is an AI prediction tool for educational purposes only.")
    print("   Always consult qualified healthcare professionals for medical decisions.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Breast Cancer Classification from Image"
    )
    parser.add_argument("image_path", help="Path to the mammogram image file")
    parser.add_argument(
        "--model", default="quick_enhanced_model.pt", help="Path to the trained model"
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image file not found: {args.image_path}")
        return

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return

    # Initialize predictor
    predictor = BreastCancerPredictor(args.model)

    if predictor.model is None:
        print("❌ Failed to load the model.")
        return

    # Make prediction
    print(f"🔍 Analyzing image: {args.image_path}")
    prediction = predictor.predict_image(args.image_path)

    if prediction:
        display_prediction(args.image_path, prediction)
    else:
        print("❌ Failed to analyze the image.")


if __name__ == "__main__":
    main()
