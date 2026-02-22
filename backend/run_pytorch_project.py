#!/usr/bin/env python3
"""
Run Breast Cancer Classification Project - PyTorch Version
Uses existing trained model to demonstrate the complete pipeline
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import models


class BreastCancerClassifier:
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

            # Load trained weights - the checkpoint contains the state dict directly
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

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True

    def predict_image(self, image_path):
        """Predict class for a single image"""
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
            }

        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None

    def evaluate_dataset(
        self, csv_file="data/test_enhanced.csv", images_dir="data/images"
    ):
        """Evaluate model on test dataset using CSV labels"""
        print(f"\n🔍 Evaluating model on dataset: {csv_file}")

        # Load CSV file
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} samples from {csv_file}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return 0.0

        results = []
        true_labels = []
        predicted_labels = []

        # Process each sample
        for idx, row in df.iterrows():
            if idx >= 100:  # Limit to 100 samples for demo
                break

            filename = row["filename"]
            true_class = row["label"]
            image_path = Path(images_dir) / filename

            if not image_path.exists():
                continue

            prediction = self.predict_image(image_path)
            if prediction:
                results.append(
                    {
                        "image": filename,
                        "true_class": true_class,
                        "predicted_class": prediction["class"],
                        "confidence": prediction["confidence"],
                    }
                )
                true_labels.append(true_class)
                predicted_labels.append(prediction["class"])

        # Calculate metrics
        if results:
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f"\n📊 Evaluation Results:")
            print(f"   Samples processed: {len(results)}")
            print(f"   Accuracy: {accuracy:.4f}")

            # Classification report
            print(f"\n📋 Classification Report:")
            print(
                classification_report(
                    true_labels, predicted_labels, target_names=self.class_names
                )
            )

            # Confusion matrix
            cm = confusion_matrix(
                true_labels, predicted_labels, labels=self.class_names
            )
            self.plot_confusion_matrix(cm, self.class_names)

            # Save results
            df_results = pd.DataFrame(results)
            df_results.to_csv("evaluation_results.csv", index=False)
            print(f"\n💾 Results saved to: evaluation_results.csv")

            return accuracy
        else:
            print("❌ No images found for evaluation")
            return 0.0

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("📈 Confusion matrix saved as: confusion_matrix.png")

    def demo_prediction(self):
        """Demonstrate prediction on a sample image"""
        print(f"\n🎯 Demo Prediction")

        # Look for a sample image in the dataset
        images_dir = Path("data/images")
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(
                images_dir.glob("*.jpg")
            )
            if image_files:
                sample_image = image_files[0]
                print(f"📸 Using sample image: {sample_image.name}")
                prediction = self.predict_image(sample_image)

                if prediction:
                    print(f"🔍 Prediction Results:")
                    print(f"   Predicted Class: {prediction['class']}")
                    print(f"   Confidence: {prediction['confidence']:.4f}")
                    print(f"   Probabilities:")
                    for i, (class_name, prob) in enumerate(
                        zip(self.class_names, prediction["probabilities"])
                    ):
                        print(f"     {class_name}: {prob:.4f}")
                else:
                    print("❌ Failed to predict image")
            else:
                print("❌ No images found in data/images directory")
        else:
            print("❌ Images directory not found: data/images")


def main():
    print("🏥 Breast Cancer Classification Project - PyTorch Version")
    print("=" * 60)

    # Check if model exists
    model_path = "quick_enhanced_model.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the trained model file exists.")
        return

    # Initialize classifier
    classifier = BreastCancerClassifier(model_path)

    # Demo prediction
    classifier.demo_prediction()

    # Evaluate on dataset
    test_csv = "data/test_enhanced.csv"
    if os.path.exists(test_csv):
        classifier.evaluate_dataset(test_csv)
    else:
        print(f"⚠️ Test CSV not found: {test_csv}")
        # Try other CSV files
        for csv_file in ["data/val_enhanced.csv", "data/train_enhanced.csv"]:
            if os.path.exists(csv_file):
                print(f"📊 Using {csv_file} for evaluation")
                classifier.evaluate_dataset(csv_file)
                break

    print("\n✅ Project execution completed!")


if __name__ == "__main__":
    main()
