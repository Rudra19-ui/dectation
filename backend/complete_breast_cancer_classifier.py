#!/usr/bin/env python3
"""
Complete Breast Cancer Classification Project
Using Transfer Learning with EfficientNetB0/ResNet50
"""

import os
import shutil
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# Configuration
RANDOM_SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
CLASS_NAMES = ["normal", "benign", "malignant"]

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class BreastCancerClassifier:
    """Complete breast cancer classification system"""

    def __init__(self, base_model="efficientnet", img_size=224, num_classes=3):
        self.base_model = base_model
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES
        self.model = None
        self.history = None

    def organize_dataset(self, source_dirs, output_dir="organized_dataset"):
        """
        Organize and label images from multiple source directories

        Args:
            source_dirs (list): List of source directory paths
            output_dir (str): Output directory for organized dataset
        """
        print("📁 Organizing dataset...")

        # Create output directories
        for class_name in self.class_names:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

        total_images = 0

        for source_dir in source_dirs:
            if not os.path.exists(source_dir):
                print(f"⚠️ Source directory not found: {source_dir}")
                continue

            print(f"Processing: {source_dir}")

            # Process files in source directory
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
                        file_path = os.path.join(root, file)

                        # Determine class from directory structure or filename
                        class_name = self._determine_class(file_path, root)

                        if class_name:
                            # Convert DICOM to PNG if needed
                            if file.lower().endswith(".dcm"):
                                try:
                                    png_path = self._convert_dicom_to_png(file_path)
                                    if png_path:
                                        file_path = png_path
                                    else:
                                        continue
                                except Exception as e:
                                    print(f"Error converting DICOM {file}: {e}")
                                    continue

                            # Copy to organized directory
                            new_filename = f"{class_name}_{total_images:04d}.png"
                            dest_path = os.path.join(
                                output_dir, class_name, new_filename
                            )

                            try:
                                shutil.copy2(file_path, dest_path)
                                total_images += 1
                                if total_images % 100 == 0:
                                    print(f"  Processed {total_images} images...")
                            except Exception as e:
                                print(f"Error copying {file}: {e}")

        print(f"✅ Dataset organized: {total_images} images")
        return output_dir

    def _determine_class(self, file_path, root_dir):
        """Determine class from file path or directory structure"""
        file_lower = file_path.lower()
        root_lower = root_dir.lower()

        # Check for class names in path
        for class_name in self.class_names:
            if class_name in file_lower or class_name in root_lower:
                return class_name

        # Check filename patterns
        filename = os.path.basename(file_path).lower()
        if "normal" in filename or "healthy" in filename:
            return "normal"
        elif "benign" in filename:
            return "benign"
        elif "malignant" in filename or "cancer" in filename:
            return "malignant"

        return None

    def _convert_dicom_to_png(self, dicom_path):
        """Convert DICOM file to PNG"""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array

            # Normalize to 0-255
            if pixel_array.max() > 255:
                pixel_array = (
                    (pixel_array - pixel_array.min())
                    / (pixel_array.max() - pixel_array.min())
                    * 255
                ).astype(np.uint8)

            # Convert to PIL Image
            image = Image.fromarray(pixel_array)

            # Save as PNG
            png_path = dicom_path.replace(".dcm", ".png")
            image.save(png_path)
            return png_path
        except Exception as e:
            print(f"Error converting DICOM: {e}")
            return None

    def preprocess_images(self, input_dir, output_dir="preprocessed_dataset"):
        """
        Preprocess images: resize, normalize, apply augmentation

        Args:
            input_dir (str): Input directory with organized images
            output_dir (str): Output directory for preprocessed images
        """
        print("🔄 Preprocessing images...")

        # Create output directories
        for class_name in self.class_names:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
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
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        total_processed = 0

        for class_name in self.class_names:
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            print(f"Processing {class_name} images...")

            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(class_dir, filename)

                    try:
                        # Load and resize image
                        image = Image.open(file_path).convert("RGB")
                        image = image.resize((self.img_size, self.img_size))

                        # Save original (for validation/test)
                        dest_path = os.path.join(output_dir, class_name, filename)
                        image.save(dest_path)
                        total_processed += 1

                        # Create augmented versions (for training)
                        if total_processed <= 1000:  # Limit augmented samples
                            img_array = np.array(image)
                            img_array = img_array.reshape((1,) + img_array.shape)

                            # Generate augmented images
                            for i, batch in enumerate(
                                train_datagen.flow(img_array, batch_size=1)
                            ):
                                if i >= 2:  # Generate 2 augmented versions per image
                                    break
                                aug_img = batch[0]
                                aug_img = (aug_img * 255).astype(np.uint8)
                                aug_img = Image.fromarray(aug_img)

                                aug_filename = (
                                    f"{os.path.splitext(filename)[0]}_aug_{i}.png"
                                )
                                aug_path = os.path.join(
                                    output_dir, class_name, aug_filename
                                )
                                aug_img.save(aug_path)

                        if total_processed % 100 == 0:
                            print(f"  Processed {total_processed} images...")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        print(f"✅ Preprocessing complete: {total_processed} images")
        return output_dir

    def split_dataset(
        self, input_dir, train_dir="train", val_dir="val", test_dir="test"
    ):
        """
        Split dataset into train/validation/test sets

        Args:
            input_dir (str): Input directory with preprocessed images
            train_dir (str): Training directory
            val_dir (str): Validation directory
            test_dir (str): Test directory
        """
        print("📊 Splitting dataset...")

        # Create split directories
        for split_dir in [train_dir, val_dir, test_dir]:
            for class_name in self.class_names:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

        total_images = 0

        for class_name in self.class_names:
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            # Get all images for this class
            images = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            if len(images) == 0:
                continue

            print(f"Splitting {class_name}: {len(images)} images")

            # Split: 70% train, 15% val, 15% test
            train_imgs, temp_imgs = train_test_split(
                images, test_size=0.3, random_state=RANDOM_SEED
            )
            val_imgs, test_imgs = train_test_split(
                temp_imgs, test_size=0.5, random_state=RANDOM_SEED
            )

            # Copy images to respective directories
            for img_list, dest_dir in [
                (train_imgs, train_dir),
                (val_imgs, val_dir),
                (test_imgs, test_dir),
            ]:
                for img in img_list:
                    src_path = os.path.join(class_dir, img)
                    dst_path = os.path.join(dest_dir, class_name, img)
                    shutil.copy2(src_path, dst_path)
                    total_images += 1

        print(f"✅ Dataset split complete: {total_images} images")

        # Print statistics
        for split_name, split_dir in [
            ("Train", train_dir),
            ("Validation", val_dir),
            ("Test", test_dir),
        ]:
            print(f"\n{split_name} set:")
            for class_name in self.class_names:
                class_path = os.path.join(split_dir, class_name)
                if os.path.exists(class_path):
                    count = len(
                        [
                            f
                            for f in os.listdir(class_path)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        ]
                    )
                    print(f"  {class_name}: {count} images")

    def build_model(self, base_model="efficientnet"):
        """
        Build transfer learning model

        Args:
            base_model (str): 'efficientnet' or 'resnet50'
        """
        print(f"🔧 Building {base_model} model...")

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

        # Validation/test preprocessing
        self.val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Load base model
        if base_model == "efficientnet":
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
        return self.model

    def train_model(self, train_dir, val_dir, epochs=EPOCHS):
        """
        Train the model

        Args:
            train_dir (str): Training data directory
            val_dir (str): Validation data directory
            epochs (int): Number of training epochs
        """
        print("🚀 Training model...")

        # Create data generators
        train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )

        val_generator = self.val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                "best_model.h5", monitor="val_accuracy", save_best_only=True
            ),
        ]

        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
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
        ax1.plot(self.history.history["accuracy"], label="Train Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history["loss"], label="Train Loss")
        ax2.plot(self.history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_filename = f"training_history_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Training history saved as: {plot_filename}")
        plt.show()

    def evaluate_model(self, test_dir):
        """
        Evaluate model on test set

        Args:
            test_dir (str): Test data directory
        """
        print("🧪 Evaluating model...")

        # Create test generator
        test_generator = self.val_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)

        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

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
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        # Save confusion matrix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cm_filename = f"confusion_matrix_{timestamp}.png"
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

    def save_model(self, filepath="breast_cancer_model.h5"):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"✅ Model saved as: {filepath}")

    def load_model(self, filepath="breast_cancer_model.h5"):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from: {filepath}")

    def predict_image(self, image_path):
        """
        Predict class for a single image

        Args:
            image_path (str): Path to image file

        Returns:
            dict: Prediction results
        """
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
        }

    def suggest_improvements(self):
        """Suggest improvements to boost accuracy"""
        print("\n💡 SUGGESTIONS TO IMPROVE ACCURACY:")
        print("=" * 50)

        suggestions = [
            "1. Data Augmentation:",
            "   - Add more augmentation techniques (elastic deformation, noise)",
            "   - Use mixup or cutmix augmentation",
            "   - Implement class-specific augmentation",
            "",
            "2. Model Architecture:",
            "   - Try different base models (EfficientNetB1-B7, ResNet101)",
            "   - Use ensemble of multiple models",
            "   - Implement attention mechanisms",
            "",
            "3. Training Strategy:",
            "   - Use learning rate scheduling",
            "   - Implement gradient clipping",
            "   - Use focal loss for class imbalance",
            "   - Try different optimizers (AdamW, RAdam)",
            "",
            "4. Data Quality:",
            "   - Collect more training data",
            "   - Clean and validate dataset quality",
            "   - Use data cleaning techniques",
            "",
            "5. Advanced Techniques:",
            "   - Implement transfer learning with fine-tuning",
            "   - Use pre-trained medical imaging models",
            "   - Implement cross-validation",
            "   - Use advanced regularization techniques",
        ]

        for suggestion in suggestions:
            print(suggestion)


def main():
    """Main function to run the complete pipeline"""
    print("🏥 Complete Breast Cancer Classification Project")
    print("=" * 60)

    # Initialize classifier
    classifier = BreastCancerClassifier(base_model="efficientnet")

    # Define source directories (update these paths)
    source_dirs = [
        "data/images",  # Your existing dataset
        "dataset4/archive",  # Additional dataset
    ]

    # Step 1: Organize dataset
    print("\n📁 Step 1: Organizing dataset...")
    organized_dir = classifier.organize_dataset(source_dirs)

    # Step 2: Preprocess images
    print("\n🔄 Step 2: Preprocessing images...")
    preprocessed_dir = classifier.preprocess_images(organized_dir)

    # Step 3: Split dataset
    print("\n📊 Step 3: Splitting dataset...")
    classifier.split_dataset(preprocessed_dir)

    # Step 4: Build model
    print("\n🔧 Step 4: Building model...")
    classifier.build_model()

    # Step 5: Train model
    print("\n🚀 Step 5: Training model...")
    classifier.train_model("train", "val", epochs=30)

    # Step 6: Plot training history
    print("\n📈 Step 6: Plotting training history...")
    classifier.plot_training_history()

    # Step 7: Evaluate model
    print("\n🧪 Step 7: Evaluating model...")
    results = classifier.evaluate_model("test")

    # Step 8: Save model
    print("\n💾 Step 8: Saving model...")
    classifier.save_model()

    # Step 9: Test prediction
    print("\n🔍 Step 9: Testing prediction...")
    # Find a test image for prediction
    test_images = []
    for class_name in classifier.class_names:
        test_dir = os.path.join("test", class_name)
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
        print(f"\nPrediction for {os.path.basename(test_images[0])}:")
        print(f"Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")

    # Step 10: Suggest improvements
    print("\n💡 Step 10: Improvement suggestions...")
    classifier.suggest_improvements()

    print("\n🎉 Complete pipeline finished!")


if __name__ == "__main__":
    main()
