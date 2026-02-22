#!/usr/bin/env python3
"""
Master Breast Cancer Classification Script
Complete pipeline: Load dataset → Train model → Predict → Evaluate
Optimized for Windows compatibility with proper function structure
"""

import multiprocessing
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress warnings
warnings.filterwarnings("ignore")

# Windows-specific TensorFlow configuration will be set in main block


class BreastCancerMasterClassifier:
    """
    Master classifier for breast cancer detection with complete pipeline
    """

    def __init__(
        self,
        dataset_path="E:\\rudra\\project\\dataset",
        model_path="breast_cancer_master_model.h5",
    ):
        """
        Initialize the master classifier

        Args:
            dataset_path (str): Path to dataset folder
            model_path (str): Path to save/load model
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path
        self.model = None
        self.class_names = ["benign", "malignant", "normal"]
        self.input_size = (224, 224)
        self.batch_size = 8  # Windows-safe batch size

        # Training configuration
        self.epochs = 30
        self.learning_rate = 1e-4
        self.validation_split = 0.2

        # Statistics
        self.training_history = None
        self.test_accuracy = None
        self.test_predictions = None
        self.test_true_labels = None

        print("🏥 Breast Cancer Master Classifier Initialized")
        print(f"📁 Dataset path: {self.dataset_path}")
        print(f"📁 Model path: {self.model_path}")

    def configure_tensorflow(self):
        """
        Configure TensorFlow for Windows compatibility
        """
        print("🔧 Configuring TensorFlow for Windows...")

        # Configure GPU memory growth (if using GPU)
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            else:
                print("💻 Using CPU for training")
        except Exception as e:
            print(f"⚠️ GPU configuration failed: {e}")

        # Set thread configuration for single-threaded operation
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Disable TensorFlow logging
        tf.get_logger().setLevel("ERROR")

        print(f"✅ TensorFlow version: {tf.__version__}")

    def verify_dataset_structure(self):
        """
        Verify that dataset has the correct structure

        Returns:
            bool: True if dataset structure is correct
        """
        print("🔍 Verifying dataset structure...")

        if not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return False

        # Check for required class folders
        class_folders = ["benign", "malignant", "normal"]
        missing_folders = []
        total_images = 0

        for class_name in class_folders:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                missing_folders.append(class_name)
            else:
                # Count images in class folder
                image_files = list(class_path.glob("*.jpg")) + list(
                    class_path.glob("*.png")
                )
                count = len(image_files)
                total_images += count
                print(f"✅ {class_name}: {count} images")

        if missing_folders:
            print(f"❌ Missing class folders: {missing_folders}")
            return False

        if total_images == 0:
            print("❌ No images found in dataset")
            return False

        print(f"✅ Dataset structure verified: {total_images} total images")
        return True

    def create_data_generators(self):
        """
        Create data generators for training and validation

        Returns:
            tuple: (train_generator, validation_generator)
        """
        print("📊 Creating data generators...")

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=self.validation_split,
        )

        # Simple rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255, validation_split=self.validation_split
        )

        # Create generators with Windows-safe settings
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            workers=0,  # CRITICAL: No multiprocessing
            use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
            max_queue_size=10,
        )

        validation_generator = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            workers=0,  # CRITICAL: No multiprocessing
            use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
            max_queue_size=10,
        )

        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        print(f"✅ Class indices: {train_generator.class_indices}")

        return train_generator, validation_generator

    def build_model(self):
        """
        Build the CNN model using transfer learning

        Returns:
            keras.Model: Compiled model
        """
        print("🏗️ Building CNN model...")

        # Load pre-trained ResNet50V2
        base_model = ResNet50V2(
            weights="imagenet", include_top=False, input_shape=(*self.input_size, 3)
        )

        # Freeze base model layers
        base_model.trainable = False

        # Create model
        model = keras.Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                Dense(256, activation="relu"),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(len(self.class_names), activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"✅ Model built successfully")
        print(f"📊 Total parameters: {model.count_params():,}")

        return model

    def train_model(self, train_generator, validation_generator):
        """
        Train the model

        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
        """
        print("🚀 Starting model training...")

        # Build model
        self.model = self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            ModelCheckpoint(
                self.model_path, monitor="val_accuracy", save_best_only=True, verbose=1
            ),
        ]

        # Train model with Windows-safe settings
        self.training_history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1,
            workers=0,  # CRITICAL: No multiprocessing
            use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
            max_queue_size=10,
        )

        print("✅ Model training completed!")
        print(f"📁 Model saved to: {self.model_path}")

    def load_or_train_model(self):
        """
        Load existing model or train new one

        Returns:
            bool: True if model is ready
        """
        print("🔍 Checking for existing model...")

        if os.path.exists(self.model_path):
            try:
                print(f"📂 Loading existing model: {self.model_path}")
                self.model = load_model(self.model_path)
                print("✅ Model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Will train new model...")

        # Train new model
        print("🔄 Training new model...")

        # Verify dataset
        if not self.verify_dataset_structure():
            return False

        # Create data generators
        train_generator, validation_generator = self.create_data_generators()

        # Train model
        self.train_model(train_generator, validation_generator)

        return True

    def predict_sample_images(self, num_samples=5):
        """
        Predict on sample images from each class

        Args:
            num_samples (int): Number of samples per class
        """
        print(f"🔮 Predicting on {num_samples} sample images per class...")

        predictions = []
        true_labels = []
        sample_images = []

        for class_name in self.class_names:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue

            # Get sample images from this class
            image_files = list(class_path.glob("*.jpg")) + list(
                class_path.glob("*.png")
            )
            if not image_files:
                continue

            # Randomly select samples
            selected_files = random.sample(
                image_files, min(num_samples, len(image_files))
            )

            for image_file in selected_files:
                try:
                    # Load and preprocess image
                    img = tf.keras.preprocessing.image.load_img(
                        image_file, target_size=self.input_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Make prediction
                    pred = self.model.predict(img_array, verbose=0)
                    predicted_class = self.class_names[np.argmax(pred[0])]
                    confidence = np.max(pred[0])

                    # Store results
                    predictions.append(predicted_class)
                    true_labels.append(class_name)
                    sample_images.append(
                        {
                            "file": image_file.name,
                            "true_class": class_name,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "probabilities": pred[0],
                        }
                    )

                    print(
                        f"📄 {image_file.name}: {class_name} → {predicted_class} ({confidence:.3f})"
                    )

                except Exception as e:
                    print(f"❌ Error predicting {image_file.name}: {e}")

        return predictions, true_labels, sample_images

    def evaluate_model(self, predictions, true_labels):
        """
        Evaluate model performance

        Args:
            predictions (list): Predicted classes
            true_labels (list): True classes
        """
        print("📊 Evaluating model performance...")

        # Calculate accuracy
        self.test_accuracy = accuracy_score(true_labels, predictions)

        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=self.class_names)

        # Create classification report
        report = classification_report(
            true_labels, predictions, target_names=self.class_names
        )

        # Store results
        self.test_predictions = predictions
        self.test_true_labels = true_labels

        # Print results
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy: {self.test_accuracy:.4f} ({self.test_accuracy*100:.2f}%)")

        print(f"\n📋 Classification Report:")
        print(report)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        return self.test_accuracy, cm, report

    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix

        Args:
            cm (numpy.ndarray): Confusion matrix
        """
        print("📊 Plotting confusion matrix...")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix - Breast Cancer Classification")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()

        # Save plot
        plot_filename = (
            f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"📁 Confusion matrix saved as: {plot_filename}")
        plt.show()

    def plot_training_history(self):
        """
        Plot training history if available
        """
        if self.training_history is None:
            print("⚠️ No training history available")
            return

        print("📊 Plotting training history...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(self.training_history.history["accuracy"], label="Training Accuracy")
        ax1.plot(
            self.training_history.history["val_accuracy"], label="Validation Accuracy"
        )
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.training_history.history["loss"], label="Training Loss")
        ax2.plot(self.training_history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        plot_filename = (
            f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"📁 Training history saved as: {plot_filename}")
        plt.show()

    def save_results(self, sample_images, accuracy, cm, report):
        """
        Save results to files

        Args:
            sample_images (list): Sample image predictions
            accuracy (float): Test accuracy
            cm (numpy.ndarray): Confusion matrix
            report (str): Classification report
        """
        print("💾 Saving results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions to CSV
        predictions_df = pd.DataFrame(sample_images)
        csv_filename = f"predictions_{timestamp}.csv"
        predictions_df.to_csv(csv_filename, index=False)
        print(f"📁 Predictions saved as: {csv_filename}")

        # Save confusion matrix to CSV
        cm_df = pd.DataFrame(cm, columns=self.class_names, index=self.class_names)
        cm_filename = f"confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(cm_filename)
        print(f"📁 Confusion matrix saved as: {cm_filename}")

        # Save classification report
        report_filename = f"classification_report_{timestamp}.txt"
        with open(report_filename, "w") as f:
            f.write(f"Breast Cancer Classification Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write(report)
        print(f"📁 Classification report saved as: {report_filename}")

        # Save summary
        summary_filename = f"summary_{timestamp}.txt"
        with open(summary_filename, "w") as f:
            f.write(f"Breast Cancer Classification Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write(f"Input size: {self.input_size}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Total predictions: {len(sample_images)}\n")
        print(f"📁 Summary saved as: {summary_filename}")

    def run_complete_pipeline(self, num_samples=5):
        """
        Run the complete pipeline: Load → Train → Predict → Evaluate

        Args:
            num_samples (int): Number of sample images to predict
        """
        print("=" * 80)
        print("🏥 Breast Cancer Classification - Complete Pipeline")
        print("=" * 80)

        # Step 1: Configure TensorFlow
        self.configure_tensorflow()

        # Step 2: Load or train model
        if not self.load_or_train_model():
            print("❌ Failed to load or train model")
            return False

        # Step 3: Predict on sample images
        predictions, true_labels, sample_images = self.predict_sample_images(
            num_samples
        )

        if not predictions:
            print("❌ No predictions made")
            return False

        # Step 4: Evaluate model
        accuracy, cm, report = self.evaluate_model(predictions, true_labels)

        # Step 5: Plot training history (if available)
        self.plot_training_history()

        # Step 6: Save results
        self.save_results(sample_images, accuracy, cm, report)

        # Step 7: Print final summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📁 Model saved: {self.model_path}")
        print(f"📊 Predictions made: {len(sample_images)}")
        print(f"📁 Results saved with timestamp")

        return True


if __name__ == "__main__":
    """
    Main execution block - prevents multiprocessing issues on Windows
    """
    # CRITICAL: Set environment variables BEFORE any other code execution
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # CRITICAL: Set multiprocessing start method to 'spawn'
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("✅ Multiprocessing start method set to 'spawn'")
    except RuntimeError as e:
        print(f"⚠️ Could not set start method: {e}")

    print("🏥 Breast Cancer Classification Master Script")
    print("Complete pipeline: Load → Train → Predict → Evaluate")
    print("Optimized for Windows compatibility")

    # Create master classifier
    classifier = BreastCancerMasterClassifier()

    # Run complete pipeline
    success = classifier.run_complete_pipeline(num_samples=5)

    if success:
        print("\n🎉 All tasks completed successfully!")
        print("📁 Check the generated files for detailed results")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

    # Keep the process alive to show results
    input("\nPress Enter to exit...")
