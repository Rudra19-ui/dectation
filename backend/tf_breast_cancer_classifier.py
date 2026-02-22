#!/usr/bin/env python3
"""
TensorFlow Breast Cancer Classification Project
Avoids multiprocessing issues on Windows
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
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# Configuration
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch size to avoid memory issues
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
CLASS_NAMES = ["normal", "benign", "malignant"]

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Disable multiprocessing for Windows compatibility
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TFBreastCancerClassifier:
    """TensorFlow-based breast cancer classification system"""

    def __init__(self, base_model="efficientnet"):
        self.base_model = base_model
        self.img_size = IMG_SIZE
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        self.model = None
        self.history = None

    def prepare_dataset(self, source_dir="split_dataset"):
        """Prepare dataset from existing split_dataset structure"""
        print("📁 Preparing dataset...")

        # Check if split_dataset exists
        if not os.path.exists(source_dir):
            print(f"❌ Source directory not found: {source_dir}")
            print("Please run the dataset organization scripts first.")
            return False

        # Verify class directories exist
        for class_name in self.class_names:
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Class directory not found: {class_dir}")
                return False

        print("✅ Dataset structure verified")
        return True

    def build_model(self):
        """Build transfer learning model"""
        print(f"🔧 Building {self.base_model} model...")

        # Load base model
        if self.base_model == "efficientnet":
            base = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3),
            )
        else:  # resnet50
            base = ResNet50(
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
        print(f"Model summary:")
        self.model.summary()
        return self.model

    def create_data_generators(
        self,
        train_dir="split_dataset/train",
        val_dir="split_dataset/val",
        test_dir="split_dataset/test",
    ):
        """Create data generators for training, validation, and testing"""
        print("🔄 Creating data generators...")

        # Data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        # Validation/test preprocessing (no augmentation)
        self.val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators with multiprocessing disabled
        self.train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
            workers=0,  # Disable multiprocessing
            use_multiprocessing=False,
        )

        self.val_generator = self.val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
            workers=0,  # Disable multiprocessing
            use_multiprocessing=False,
        )

        self.test_generator = self.val_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
            workers=0,  # Disable multiprocessing
            use_multiprocessing=False,
        )

        print("✅ Data generators created")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")

    def train_model(self, epochs=EPOCHS):
        """Train the model"""
        print("🚀 Training model...")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            ModelCheckpoint(
                "best_breast_cancer_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        print("✅ Training complete")
        return self.history

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(self.history.history["accuracy"], label="Train Accuracy", linewidth=2)
        ax1.plot(
            self.history.history["val_accuracy"],
            label="Validation Accuracy",
            linewidth=2,
        )
        ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(self.history.history["loss"], label="Train Loss", linewidth=2)
        ax2.plot(self.history.history["val_loss"], label="Validation Loss", linewidth=2)
        ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_filename = f"training_history_{self.base_model}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Training history saved as: {plot_filename}")
        plt.show()

    def evaluate_model(self):
        """Evaluate model on test set"""
        print("🧪 Evaluating model...")

        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)

        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes

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

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(y_true, y_pred, average=None)
        )

        print(f"\n📋 Per-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name.upper()}:")
            print(f"     Precision: {precision_per_class[i]:.4f}")
            print(f"     Recall: {recall_per_class[i]:.4f}")
            print(f"     F1 Score: {f1_per_class[i]:.4f}")

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
        plt.title(
            f"Confusion Matrix - {self.base_model.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.tight_layout()

        # Save confusion matrix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cm_filename = f"confusion_matrix_{self.base_model}_{timestamp}.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Confusion matrix saved as: {cm_filename}")
        plt.show()

        return {
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
        }

    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = f"breast_cancer_model_{self.base_model}_{timestamp}.h5"

        self.model.save(filepath)
        print(f"✅ Model saved as: {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from: {filepath}")

    def predict_image(self, image_path):
        """Predict class for a single image"""
        try:
            # Load and preprocess image
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(self.img_size, self.img_size)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

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

    def suggest_improvements(self):
        """Suggest improvements to boost accuracy"""
        print("\n💡 SUGGESTIONS TO IMPROVE ACCURACY:")
        print("=" * 50)

        suggestions = [
            "1. Data Augmentation:",
            "   - Add elastic deformation",
            "   - Implement mixup/cutmix augmentation",
            "   - Use class-specific augmentation",
            "   - Add noise and blur augmentation",
            "",
            "2. Model Architecture:",
            "   - Try EfficientNetB1-B7 for better performance",
            "   - Use ensemble of multiple models",
            "   - Implement attention mechanisms",
            "   - Add more dense layers",
            "",
            "3. Training Strategy:",
            "   - Implement fine-tuning (unfreeze base layers)",
            "   - Use focal loss for class imbalance",
            "   - Try different optimizers (AdamW, RAdam)",
            "   - Implement gradient clipping",
            "",
            "4. Data Quality:",
            "   - Collect more training data",
            "   - Clean and validate dataset",
            "   - Use data cleaning techniques",
            "   - Implement cross-validation",
            "",
            "5. Advanced Techniques:",
            "   - Use pre-trained medical imaging models",
            "   - Implement transfer learning with fine-tuning",
            "   - Use advanced regularization",
            "   - Implement learning rate scheduling",
        ]

        for suggestion in suggestions:
            print(suggestion)


def main():
    """Main function to run the complete pipeline"""
    print("🏥 TensorFlow Breast Cancer Classification Project")
    print("=" * 60)

    # Initialize classifier
    classifier = TFBreastCancerClassifier(base_model="efficientnet")

    # Step 1: Prepare dataset
    print("\n📁 Step 1: Preparing dataset...")
    if not classifier.prepare_dataset():
        print("❌ Dataset preparation failed. Exiting.")
        return

    # Step 2: Create data generators
    print("\n🔄 Step 2: Creating data generators...")
    classifier.create_data_generators()

    # Step 3: Build model
    print("\n🔧 Step 3: Building model...")
    classifier.build_model()

    # Step 4: Train model
    print("\n🚀 Step 4: Training model...")
    classifier.train_model(epochs=30)

    # Step 5: Plot training history
    print("\n📈 Step 5: Plotting training history...")
    classifier.plot_training_history()

    # Step 6: Evaluate model
    print("\n🧪 Step 6: Evaluating model...")
    results = classifier.evaluate_model()

    # Step 7: Save model
    print("\n💾 Step 7: Saving model...")
    model_path = classifier.save_model()

    # Step 8: Test prediction
    print("\n🔍 Step 8: Testing prediction...")
    # Find a test image for prediction
    test_images = []
    for class_name in classifier.class_names:
        test_dir = os.path.join("split_dataset/test", class_name)
        if os.path.exists(test_dir):
            images = [
                f
                for f in os.listdir(test_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if images:
                test_images.append(os.path.join(test_dir, images[0]))
                break

    if test_images:
        result = classifier.predict_image(test_images[0])
        if result["success"]:
            print(f"\nPrediction for {os.path.basename(test_images[0])}:")
            print(f"Class: {result['predicted_class'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities:")
            for i, (class_name, prob) in enumerate(
                zip(result["class_names"], result["probabilities"])
            ):
                print(f"  {class_name}: {prob:.2%}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    # Step 9: Suggest improvements
    print("\n💡 Step 9: Improvement suggestions...")
    classifier.suggest_improvements()

    print("\n🎉 Complete pipeline finished!")
    print(f"\n📋 Summary:")
    print(f"Model: {classifier.base_model.upper()}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"F1 Score: {results['f1']:.2%}")
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
