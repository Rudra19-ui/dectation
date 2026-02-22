#!/usr/bin/env python3
"""
Windows-Optimized Breast Cancer Classification Training Script
Specifically designed to avoid TensorFlow/Python freezing on Windows
Uses ImageDataGenerator safely without multiprocessing
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Windows-specific configurations
def configure_for_windows():
    """
    Configure TensorFlow and system settings for Windows compatibility
    """
    print("🔧 Configuring for Windows compatibility...")

    # Disable multiprocessing
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set memory growth for GPU (if available)
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("ℹ️ No GPU detected, using CPU")
    except Exception as e:
        print(f"⚠️ GPU configuration failed: {e}")

    # Set thread configuration
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    print("✅ Thread configuration set for single-threaded operation")

    # Disable TensorFlow logging
    tf.get_logger().setLevel("ERROR")
    print("✅ TensorFlow logging disabled")


# Configuration optimized for Windows
DATASET_PATH = r"E:\rudra\project\dataset"
MODEL_SAVE_PATH = "breast_cancer_model_windows.h5"
INPUT_SIZE = (224, 224)
BATCH_SIZE = 8  # Reduced batch size for Windows
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 3
CLASS_NAMES = ["benign", "malignant", "normal"]


def create_windows_safe_data_generators():
    """
    Create data generators optimized for Windows (no multiprocessing)
    """
    print("📂 Creating Windows-safe data generators...")

    # Data augmentation for training (reduced for stability)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,  # Reduced from 20
        width_shift_range=0.1,  # Reduced from 0.2
        height_shift_range=0.1,  # Reduced from 0.2
        shear_range=0.1,  # Reduced from 0.2
        zoom_range=0.1,  # Reduced from 0.2
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=VALIDATION_SPLIT
    )

    # Load training data with Windows-safe settings
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        workers=0,  # CRITICAL: No multiprocessing
        use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
        max_queue_size=10,  # Reduced queue size
    )

    # Load validation data with Windows-safe settings
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        workers=0,  # CRITICAL: No multiprocessing
        use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
        max_queue_size=10,  # Reduced queue size
    )

    print(f"✅ Windows-safe data generators created!")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")
    print(f"📊 Classes: {train_generator.class_indices}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔧 Workers: 0 (multiprocessing disabled)")

    return train_generator, val_generator


def create_windows_optimized_model():
    """
    Create CNN model optimized for Windows training
    """
    print("🔧 Creating Windows-optimized CNN model...")

    # Base model (ResNet50V2)
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create model with reduced complexity for Windows
    model = keras.Sequential(
        [
            # Data preprocessing
            layers.Rescaling(1.0 / 255, input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)),
            # Base model
            base_model,
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            # Reduced dropout for stability
            layers.Dropout(0.3),  # Reduced from 0.5
            # Simplified dense layers
            layers.Dense(256, activation="relu"),  # Reduced from 512
            layers.Dropout(0.2),  # Reduced from 0.3
            layers.Dense(128, activation="relu"),  # Reduced from 256
            layers.Dropout(0.1),  # Reduced from 0.2
            # Output layer
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # Compile model with conservative settings
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=1.0
        ),  # Added gradient clipping
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],  # Simplified metrics
    )

    print("✅ Windows-optimized model created successfully!")
    print(f"📊 Model summary:")
    model.summary()

    return model


def create_windows_safe_callbacks():
    """
    Create training callbacks optimized for Windows
    """
    callbacks = [
        # Early stopping with conservative patience
        EarlyStopping(
            monitor="val_loss",
            patience=15,  # Increased from 10
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate reduction with conservative settings
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,  # Less aggressive than 0.5
            patience=8,  # Increased from 5
            min_lr=1e-7,
            verbose=1,
        ),
        # Model checkpoint
        ModelCheckpoint(
            filepath="best_model_windows.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    return callbacks


def train_model_windows_safe(model, train_generator, val_generator):
    """
    Train model with Windows-safe settings
    """
    print("\n🚀 Starting Windows-safe training...")
    print(f"⏱️ Training for {EPOCHS} epochs...")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔧 Workers: 0 (multiprocessing disabled)")

    # Create callbacks
    callbacks = create_windows_safe_callbacks()

    # Train model with conservative settings
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
        workers=0,  # CRITICAL: No multiprocessing
        use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
        max_queue_size=10,  # Reduced queue size
        shuffle=True,
    )

    return history


def plot_training_history_windows(history):
    """
    Plot training history with Windows-safe settings
    """
    print("📈 Plotting training history...")

    # Create figure with reduced size for Windows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Training Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax2.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"training_history_windows_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Training history saved as: {plot_filename}")
    plt.show()


def evaluate_model_windows_safe(model, val_generator):
    """
    Evaluate model with Windows-safe settings
    """
    print("\n🔍 Evaluating model with Windows-safe settings...")

    # Reset generator
    val_generator.reset()

    # Get predictions with Windows-safe settings
    predictions = model.predict(
        val_generator,
        workers=0,  # CRITICAL: No multiprocessing
        use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
        verbose=1,
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix - Windows Training", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)

    # Save confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_filename = f"confusion_matrix_windows_{timestamp}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {cm_filename}")
    plt.show()

    # Calculate metrics
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\n📊 Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy


def save_model_windows_safe(model, save_path=None):
    """
    Save model with Windows-safe settings
    """
    if save_path is None:
        save_path = MODEL_SAVE_PATH

    print(f"\n💾 Saving model to {save_path}...")

    try:
        model.save(save_path)
        print(f"✅ Model saved successfully to {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        # Try alternative save method
        try:
            model.save_weights(save_path.replace(".h5", "_weights.h5"))
            print(f"✅ Model weights saved as alternative")
            return save_path.replace(".h5", "_weights.h5")
        except Exception as e2:
            print(f"❌ Alternative save also failed: {e2}")
            return None


def main():
    """
    Main function for Windows-optimized training
    """
    print("=" * 80)
    print("🏥 Windows-Optimized Breast Cancer Classification Training")
    print("=" * 80)
    print("🔧 Optimized for Windows compatibility")
    print("🚫 No multiprocessing (prevents freezing)")
    print("📦 Reduced batch size for stability")
    print("=" * 80)

    # Configure for Windows
    configure_for_windows()

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please ensure the dataset folder exists with the following structure:")
        print(f"  {DATASET_PATH}/")
        print("  ├── benign/")
        print("  ├── malignant/")
        print("  └── normal/")
        return

    # Check dataset structure
    expected_folders = ["benign", "malignant", "normal"]
    for folder in expected_folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Missing folder: {folder_path}")
            return
        else:
            files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            print(f"✅ {folder}: {len(files)} images")

    try:
        # Create Windows-safe data generators
        train_generator, val_generator = create_windows_safe_data_generators()

        # Create Windows-optimized model
        model = create_windows_optimized_model()

        # Train model with Windows-safe settings
        history = train_model_windows_safe(model, train_generator, val_generator)

        # Plot training history
        plot_training_history_windows(history)

        # Evaluate model
        accuracy = evaluate_model_windows_safe(model, val_generator)

        # Save model
        model_path = save_model_windows_safe(model)

        # Print final summary
        print("\n" + "=" * 80)
        print("🎉 Windows Training Complete!")
        print("=" * 80)
        print(f"📁 Model saved: {model_path}")
        print(f"📊 Final validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(
            f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}"
        )
        print(f"📊 Best validation loss: {min(history.history['val_loss']):.4f}")
        print("\n🚀 Next steps:")
        print("1. Test the model on new images")
        print("2. Fine-tune hyperparameters if needed")
        print("3. Deploy the model for inference")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Reduce batch size further (try BATCH_SIZE = 4)")
        print("2. Reduce image size (try INPUT_SIZE = (128, 128))")
        print("3. Close other applications to free memory")
        print("4. Restart Python and try again")
        print("5. Check if dataset is corrupted")


if __name__ == "__main__":
    main()
