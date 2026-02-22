#!/usr/bin/env python3
"""
Breast Cancer Classification CNN Training Script
Using TensorFlow with ImageDataGenerator and validation_split
Optimized for Windows compatibility (no multiprocessing)
"""

import os
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

# Configuration
DATASET_PATH = r"E:\rudra\project\dataset"
MODEL_SAVE_PATH = "breast_cancer_model.h5"
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 3
CLASS_NAMES = ["benign", "malignant", "normal"]

# Disable multiprocessing for Windows compatibility
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_model():
    """
    Create CNN model using transfer learning with ResNet50V2
    """
    print("🔧 Creating CNN model...")

    # Base model (ResNet50V2)
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create model
    model = keras.Sequential(
        [
            # Data preprocessing
            layers.Rescaling(1.0 / 255, input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)),
            # Base model
            base_model,
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            # Dropout for regularization
            layers.Dropout(0.5),
            # Dense layers
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            # Output layer
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy", "precision", "recall"],
    )

    print("✅ Model created successfully!")
    print(f"📊 Model summary:")
    model.summary()

    return model


def create_data_generators():
    """
    Create data generators with validation_split
    """
    print("📂 Creating data generators...")

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
        validation_split=VALIDATION_SPLIT,
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=VALIDATION_SPLIT
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        workers=0,  # Disable multiprocessing
        use_multiprocessing=False,
    )

    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        workers=0,  # Disable multiprocessing
        use_multiprocessing=False,
    )

    print(f"✅ Data generators created!")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")
    print(f"📊 Classes: {train_generator.class_indices}")

    return train_generator, val_generator


def create_callbacks():
    """
    Create training callbacks
    """
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        # Model checkpoint
        ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    return callbacks


def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history["loss"], label="Training Loss")
    axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0, 1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0, 1].set_title("Model Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    axes[1, 0].plot(history.history["precision"], label="Training Precision")
    axes[1, 0].plot(history.history["val_precision"], label="Validation Precision")
    axes[1, 0].set_title("Model Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(history.history["recall"], label="Training Recall")
    axes[1, 1].plot(history.history["val_recall"], label="Validation Recall")
    axes[1, 1].set_title("Model Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_filename = f'training_history_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Training history saved as: {plot_filename}")
    plt.show()


def evaluate_model(model, val_generator):
    """
    Evaluate the trained model
    """
    print("\n🔍 Evaluating model...")

    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator, workers=0, use_multiprocessing=False)
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
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_filename = f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved as: {cm_filename}")
    plt.show()

    # Calculate metrics
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\n📊 Final Validation Accuracy: {accuracy:.4f}")


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("🏥 Breast Cancer Classification Training")
    print("=" * 60)
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"🖼️ Input size: {INPUT_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📊 Classes: {CLASS_NAMES}")
    print("=" * 60)

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

    # Create data generators
    train_generator, val_generator = create_data_generators()

    # Create model
    model = create_model()

    # Create callbacks
    callbacks = create_callbacks()

    # Train model
    print("\n🚀 Starting training...")
    print(f"⏱️ Training for {EPOCHS} epochs...")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    evaluate_model(model, val_generator)

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Model saved as: {MODEL_SAVE_PATH}")

    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"📊 Best validation loss: {min(history.history['val_loss']):.4f}")
    print("\n🚀 Next steps:")
    print("1. Test the model on new images")
    print("2. Fine-tune hyperparameters if needed")
    print("3. Deploy the model for inference")


if __name__ == "__main__":
    main()
