#!/usr/bin/env python3
"""
Simple Breast Cancer Classifier
Avoids all multiprocessing issues
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.applications import EfficientNetB0

warnings.filterwarnings("ignore")

# Configuration
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8  # Very small batch size
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
CLASS_NAMES = ["normal", "benign", "malignant"]

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class SimpleClassifier:
    """Simple breast cancer classifier without multiprocessing"""

    def __init__(self):
        self.img_size = IMG_SIZE
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        self.model = None
        self.history = None

    def load_images_manually(self, data_dir="split_dataset"):
        """Load images manually without multiprocessing"""
        print("📁 Loading images manually...")

        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return None, None, None

        # Load training data
        train_images, train_labels = self._load_class_images(
            os.path.join(data_dir, "train")
        )
        val_images, val_labels = self._load_class_images(os.path.join(data_dir, "val"))
        test_images, test_labels = self._load_class_images(
            os.path.join(data_dir, "test")
        )

        print(
            f"✅ Loaded {len(train_images)} training, {len(val_images)} validation, {len(test_images)} test images"
        )

        return (
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_labels),
        )

    def _load_class_images(self, base_dir):
        """Load images from a directory with class subdirectories"""
        images = []
        labels = []

        if not os.path.exists(base_dir):
            print(f"⚠️ Directory not found: {base_dir}")
            return images, labels

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            print(f"  Loading {class_name} images from {class_dir}...")

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        img_path = os.path.join(class_dir, filename)
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize((self.img_size, self.img_size))
                        img_array = np.array(img) / 255.0

                        images.append(img_array)
                        labels.append(class_idx)

                    except Exception as e:
                        print(f"    Error loading {filename}: {e}")

        return np.array(images), np.array(labels)

    def build_model(self):
        """Build the model"""
        print("🔧 Building EfficientNetB0 model...")

        # Load base model
        base = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
        )

        # Freeze base model layers
        base.trainable = False

        # Build complete model
        self.model = keras.Sequential(
            [
                base,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        print("✅ Model built successfully")
        return self.model

    def train_model(self, train_data, val_data):
        """Train the model"""
        print("🚀 Training model...")

        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Convert labels to categorical
        train_labels_cat = keras.utils.to_categorical(train_labels, self.num_classes)
        val_labels_cat = keras.utils.to_categorical(val_labels, self.num_classes)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
        ]

        # Train model
        self.history = self.model.fit(
            train_images,
            train_labels_cat,
            validation_data=(val_images, val_labels_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        print("✅ Training complete")
        return self.history

    def evaluate_model(self, test_data):
        """Evaluate the model"""
        print("🧪 Evaluating model...")

        test_images, test_labels = test_data
        test_labels_cat = keras.utils.to_categorical(test_labels, self.num_classes)

        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            test_images, test_labels_cat, verbose=1
        )

        # Get predictions
        predictions = self.model.predict(test_images, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_labels

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        cm = confusion_matrix(y_true, y_pred)

        # Print results
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix - EfficientNetB0", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()

        # Save confusion matrix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cm_filename = f"confusion_matrix_simple_{timestamp}.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Confusion matrix saved as: {cm_filename}")
        plt.show()

        return {
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = f"breast_cancer_model_simple_{timestamp}.h5"

        self.model.save(filepath)
        print(f"✅ Model saved as: {filepath}")
        return filepath

    def predict_image(self, image_path):
        """Predict class for a single image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            return {
                "predicted_class": self.class_names[predicted_class],
                "confidence": confidence,
                "probabilities": predictions[0].tolist(),
                "class_names": self.class_names,
                "success": True,
            }
        except Exception as e:
            return {
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": [0.0, 0.0, 0.0],
                "class_names": self.class_names,
                "success": False,
                "error": str(e),
            }


def main():
    """Main function"""
    print("🏥 Simple Breast Cancer Classification Project")
    print("=" * 60)

    # Initialize classifier
    classifier = SimpleClassifier()

    # Step 1: Load data
    print("\n📁 Step 1: Loading data...")
    train_data, val_data, test_data = classifier.load_images_manually()

    if train_data[0] is None or len(train_data[0]) == 0:
        print(
            "❌ No training data found. Please ensure split_dataset exists with train/val/test subdirectories."
        )
        return

    # Step 2: Build model
    print("\n🔧 Step 2: Building model...")
    classifier.build_model()

    # Step 3: Train model
    print("\n🚀 Step 3: Training model...")
    classifier.train_model(train_data, val_data)

    # Step 4: Evaluate model
    print("\n🧪 Step 4: Evaluating model...")
    results = classifier.evaluate_model(test_data)

    # Step 5: Save model
    print("\n💾 Step 5: Saving model...")
    model_path = classifier.save_model()

    # Step 6: Test prediction
    print("\n🔍 Step 6: Testing prediction...")
    if test_data[0] is not None and len(test_data[0]) > 0:
        # Use first test image for prediction
        test_image = test_data[0][0]
        test_image_expanded = np.expand_dims(test_image, axis=0)
        predictions = classifier.model.predict(test_image_expanded)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print(f"\nPrediction for test image:")
        print(f"Class: {classifier.class_names[predicted_class].upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities:")
        for i, (class_name, prob) in enumerate(
            zip(classifier.class_names, predictions[0])
        ):
            print(f"  {class_name}: {prob:.2%}")

    print("\n🎉 Complete pipeline finished!")
    print(f"\n📋 Summary:")
    print(f"Model: EfficientNetB0")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"F1 Score: {results['f1']:.2%}")
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
