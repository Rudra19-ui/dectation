#!/usr/bin/env python3
"""
Deep Learning Model Builder for Mammogram Classification
======================================================

This script creates CNN models using transfer learning with:
- EfficientNetB0
- ResNet50

Features:
- Input size: 224x224x3
- Output: 3 classes (normal, benign, malignant)
- Activation: softmax
- Freeze pretrained layers
- Adam optimizer with categorical crossentropy
- Support for both PyTorch and TensorFlow/Keras
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# TensorFlow/Keras imports (optional)
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, losses, metrics, optimizers
    from tensorflow.keras.applications import EfficientNetB0, ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using PyTorch only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MammogramDataset(Dataset):
    """PyTorch Dataset for mammogram images"""

    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the dataset

        Args:
            data_dir: Directory containing class subdirectories
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ["normal", "benign", "malignant"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image files
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob("*.png")) + list(
                    class_dir.glob("*.jpg")
                )
                for img_path in image_files:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        logger.info(f"Loaded {len(self.images)} images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        from PIL import Image

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class PyTorchModelBuilder:
    """PyTorch model builder with transfer learning"""

    def __init__(
        self,
        model_type: str = "resnet50",
        num_classes: int = 3,
        input_size: int = 224,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-4,
        device: str = "auto",
    ):
        """
        Initialize the PyTorch model builder

        Args:
            model_type: Type of model ('resnet50' or 'efficientnet')
            num_classes: Number of output classes
            input_size: Input image size
            freeze_backbone: Whether to freeze pretrained layers
            learning_rate: Learning rate for optimizer
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        self.learning_rate = learning_rate

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Create optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Create transforms
        self.train_transform = self._create_train_transforms()
        self.val_transform = self._create_val_transforms()

    def _create_model(self) -> nn.Module:
        """Create the model with transfer learning"""
        if self.model_type == "resnet50":
            # Load ResNet50 with pretrained weights
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            # Freeze backbone if requested
            if self.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

            # Replace the final layer
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
            )

        elif self.model_type == "efficientnet":
            # Load EfficientNet-B0 with pretrained weights
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )

            # Freeze backbone if requested
            if self.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

            # Replace the final layer
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info(f"Created {self.model_type} model with {self.num_classes} classes")
        return model

    def _create_train_transforms(self):
        """Create training transforms"""
        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _create_val_transforms(self):
        """Create validation transforms"""
        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def create_data_loaders(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Dict[str, DataLoader]:
        """Create data loaders for training, validation, and testing"""

        # Create datasets
        train_dataset = MammogramDataset(train_dir, self.train_transform)
        val_dataset = MammogramDataset(val_dir, self.val_transform)

        loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        }

        if test_dir:
            test_dataset = MammogramDataset(test_dir, self.val_transform)
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        logger.info(f"Created data loaders: {len(loaders)} splits")
        return loaders

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def save_model(self, filepath: str):
        """Save the model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
                "num_classes": self.num_classes,
                "input_size": self.input_size,
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Model loaded from {filepath}")


class TensorFlowModelBuilder:
    """TensorFlow/Keras model builder with transfer learning"""

    def __init__(
        self,
        model_type: str = "resnet50",
        num_classes: int = 3,
        input_size: int = 224,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-4,
    ):
        """
        Initialize the TensorFlow model builder

        Args:
            model_type: Type of model ('resnet50' or 'efficientnet')
            num_classes: Number of output classes
            input_size: Input image size
            freeze_backbone: Whether to freeze pretrained layers
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")

        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        self.learning_rate = learning_rate

        # Create model
        self.model = self._create_model()

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.CategoricalCrossentropy(),
            metrics=[
                metrics.CategoricalAccuracy(),
                metrics.Precision(),
                metrics.Recall(),
            ],
        )

        logger.info(f"Created {self.model_type} model with {self.num_classes} classes")

    def _create_model(self):
        """Create the model with transfer learning"""
        input_shape = (self.input_size, self.input_size, 3)

        if self.model_type == "resnet50":
            # Load ResNet50 with pretrained weights
            base_model = ResNet50(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze backbone if requested
            if self.freeze_backbone:
                base_model.trainable = False

            # Create the full model
            inputs = keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)

            model = keras.Model(inputs, outputs)

        elif self.model_type == "efficientnet":
            # Load EfficientNet-B0 with pretrained weights
            base_model = EfficientNetB0(
                weights="imagenet", include_top=False, input_shape=input_shape
            )

            # Freeze backbone if requested
            if self.freeze_backbone:
                base_model.trainable = False

            # Create the full model
            inputs = keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(512, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)

            model = keras.Model(inputs, outputs)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return model

    def create_data_generators(
        self, train_dir: str, val_dir: str, test_dir: str = None, batch_size: int = 32
    ):
        """Create data generators for training, validation, and testing"""

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
        )

        # Validation/Test data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators
        generators = {
            "train": train_datagen.flow_from_directory(
                train_dir,
                target_size=(self.input_size, self.input_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
            ),
            "val": val_datagen.flow_from_directory(
                val_dir,
                target_size=(self.input_size, self.input_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            ),
        }

        if test_dir:
            generators["test"] = val_datagen.flow_from_directory(
                test_dir,
                target_size=(self.input_size, self.input_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )

        logger.info(f"Created data generators: {len(generators)} splits")
        return generators

    def save_model(self, filepath: str):
        """Save the model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


def create_model_summary(model_builder, framework: str = "pytorch"):
    """Create a summary of the model architecture"""
    summary_path = f"model_summary_{framework}_{model_builder.model_type}.txt"

    with open(summary_path, "w") as f:
        f.write(f"MODEL SUMMARY - {framework.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Type: {model_builder.model_type}\n")
        f.write(f"Framework: {framework}\n")
        f.write(
            f"Input Size: {model_builder.input_size}x{model_builder.input_size}x3\n"
        )
        f.write(f"Output Classes: {model_builder.num_classes}\n")
        f.write(f"Freeze Backbone: {model_builder.freeze_backbone}\n")
        f.write(f"Learning Rate: {model_builder.learning_rate}\n")

        if framework == "pytorch":
            f.write(f"Device: {model_builder.device}\n")
            f.write(
                f"Total Parameters: {sum(p.numel() for p in model_builder.model.parameters())}\n"
            )
            f.write(
                f"Trainable Parameters: {sum(p.numel() for p in model_builder.model.parameters() if p.requires_grad)}\n"
            )
        else:
            f.write(f"Total Parameters: {model_builder.model.count_params()}\n")
            f.write(
                f"Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model_builder.model.trainable_weights])}\n"
            )

    logger.info(f"Model summary saved to {summary_path}")


def main():
    """Main function to build models"""
    parser = argparse.ArgumentParser(description="Deep Learning Model Builder")
    parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Deep learning framework to use",
    )
    parser.add_argument(
        "--model-type",
        choices=["resnet50", "efficientnet"],
        default="resnet50",
        help="Type of model to build",
    )
    parser.add_argument(
        "--num-classes", type=int, default=3, help="Number of output classes"
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze pretrained layers",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--train-dir", default="split_dataset/train", help="Training data directory"
    )
    parser.add_argument(
        "--val-dir", default="split_dataset/val", help="Validation data directory"
    )
    parser.add_argument(
        "--test-dir", default="split_dataset/test", help="Test data directory"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for models"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save the model after building"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Build model
    if args.framework == "pytorch":
        model_builder = PyTorchModelBuilder(
            model_type=args.model_type,
            num_classes=args.num_classes,
            input_size=args.input_size,
            freeze_backbone=args.freeze_backbone,
            learning_rate=args.learning_rate,
        )

        # Create data loaders
        data_loaders = model_builder.create_data_loaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
        )

        # Test the model with a sample batch
        logger.info("Testing model with sample batch...")
        sample_batch = next(iter(data_loaders["train"]))
        sample_images, sample_labels = sample_batch
        sample_images = sample_images.to(model_builder.device)

        with torch.no_grad():
            sample_output = model_builder.model(sample_images)
            logger.info(f"Sample output shape: {sample_output.shape}")
            logger.info(f"Sample labels shape: {sample_labels.shape}")

    else:  # TensorFlow
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not available. Please install tensorflow.")
            return

        model_builder = TensorFlowModelBuilder(
            model_type=args.model_type,
            num_classes=args.num_classes,
            input_size=args.input_size,
            freeze_backbone=args.freeze_backbone,
            learning_rate=args.learning_rate,
        )

        # Create data generators
        data_generators = model_builder.create_data_generators(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
        )

        # Test the model with a sample batch
        logger.info("Testing model with sample batch...")
        sample_batch = next(data_generators["train"])
        sample_images, sample_labels = sample_batch
        sample_output = model_builder.model.predict(sample_images)
        logger.info(f"Sample output shape: {sample_output.shape}")
        logger.info(f"Sample labels shape: {sample_labels.shape}")

    # Create model summary
    create_model_summary(model_builder, args.framework)

    # Save model if requested
    if args.save_model:
        model_filename = f"{args.framework}_{args.model_type}_model"
        model_path = output_dir / model_filename

        if args.framework == "pytorch":
            model_builder.save_model(str(model_path) + ".pt")
        else:
            model_builder.save_model(str(model_path) + ".h5")

        logger.info(f"Model saved to {model_path}")

    logger.info("Model building completed successfully!")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Framework: {args.framework}")
    logger.info(f"Classes: {args.num_classes}")
    logger.info(f"Input size: {args.input_size}x{args.input_size}x3")


if __name__ == "__main__":
    main()
