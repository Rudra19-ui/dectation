#!/usr/bin/env python3
"""
Mammogram Image Preprocessing Pipeline
=====================================

This script preprocesses mammogram images for deep learning training:
- Resize all images to 224x224
- Convert DICOM to PNG if needed
- Normalize pixel values to [0, 1]
- Apply data augmentation (rotation, zoom, flip)

Requirements:
- PyTorch
- torchvision
- albumentations
- pydicom
- PIL
- numpy
- opencv-python
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MammogramPreprocessor:
    """Comprehensive mammogram image preprocessor for deep learning"""

    def __init__(
        self,
        input_dir: str = "organized_dataset",
        output_dir: str = "preprocessed_dataset",
        target_size: Tuple[int, int] = (224, 224),
        normalize_range: Tuple[float, float] = (0.0, 1.0),
        convert_dicom: bool = True,
        apply_augmentation: bool = True,
        augmentation_factor: int = 2,
    ):
        """
        Initialize the preprocessor

        Args:
            input_dir: Directory containing organized images
            output_dir: Directory to save preprocessed images
            target_size: Target image size (width, height)
            normalize_range: Normalization range (min, max)
            convert_dicom: Whether to convert DICOM files
            apply_augmentation: Whether to apply data augmentation
            augmentation_factor: Number of augmented versions per image
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.convert_dicom = convert_dicom
        self.apply_augmentation = apply_augmentation
        self.augmentation_factor = augmentation_factor

        # Create output directories
        self._create_output_dirs()

        # Initialize transforms
        self._setup_transforms()

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "dicom_converted": 0,
            "augmented_images": 0,
            "errors": 0,
            "class_counts": {"normal": 0, "benign": 0, "malignant": 0},
        }

    def _create_output_dirs(self):
        """Create output directory structure"""
        # Main directories
        for class_name in ["normal", "benign", "malignant"]:
            (self.output_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Split directories
        for split in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                (self.output_dir / "splits" / split / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        logger.info(f"Created output directory structure in {self.output_dir}")

    def _setup_transforms(self):
        """Setup image transforms and augmentation"""
        # Basic preprocessing transforms
        self.basic_transform = A.Compose(
            [
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        # Data augmentation transforms
        self.augmentation_transform = A.Compose(
            [
                A.Resize(self.target_size[0], self.target_size[1]),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.3),
                        A.GaussianBlur(blur_limit=3, p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        logger.info("Transforms initialized")

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image from various formats (PNG, JPEG, DICOM)"""
        try:
            if image_path.suffix.lower() in [".dcm", ".dicom"]:
                if not self.convert_dicom:
                    logger.warning(f"Skipping DICOM file: {image_path}")
                    return None

                # Load DICOM file
                dcm = pydicom.dcmread(str(image_path))
                image = dcm.pixel_array

                # Convert to 8-bit if needed
                if image.dtype != np.uint8:
                    image = (
                        (image - image.min()) / (image.max() - image.min()) * 255
                    ).astype(np.uint8)

                self.stats["dicom_converted"] += 1

            else:
                # Load regular image
                image = cv2.imread(str(image_path))
                if image is None:
                    # Try with PIL
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    elif len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")

            # Ensure RGB format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            return image

        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            self.stats["errors"] += 1
            return None

    def _preprocess_image(
        self, image: np.ndarray, apply_aug: bool = False
    ) -> torch.Tensor:
        """Preprocess a single image"""
        try:
            if apply_aug:
                transformed = self.augmentation_transform(image=image)
            else:
                transformed = self.basic_transform(image=image)

            return transformed["image"]

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            self.stats["errors"] += 1
            return None

    def _save_tensor_as_image(self, tensor: torch.Tensor, output_path: Path):
        """Save tensor as image file"""
        try:
            # Convert tensor to numpy array
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = tensor * std + mean

            # Convert to numpy and transpose
            image = tensor.squeeze(0).permute(1, 2, 0).numpy()

            # Clip to valid range
            image = np.clip(image, 0, 1)

            # Convert to 8-bit
            image = (image * 255).astype(np.uint8)

            # Save as PNG
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logger.error(f"Error saving image {output_path}: {e}")
            self.stats["errors"] += 1

    def process_single_image(
        self, input_path: Path, output_path: Path, class_name: str
    ):
        """Process a single image with optional augmentation"""
        # Load image
        image = self._load_image(input_path)
        if image is None:
            return

        # Process original image
        processed = self._preprocess_image(image, apply_aug=False)
        if processed is not None:
            self._save_tensor_as_image(processed, output_path)
            self.stats["total_processed"] += 1
            self.stats["class_counts"][class_name] += 1

        # Apply augmentation if requested
        if self.apply_augmentation:
            for i in range(self.augmentation_factor):
                aug_path = (
                    output_path.parent
                    / f"{output_path.stem}_aug_{i}{output_path.suffix}"
                )
                processed_aug = self._preprocess_image(image, apply_aug=True)
                if processed_aug is not None:
                    self._save_tensor_as_image(processed_aug, aug_path)
                    self.stats["augmented_images"] += 1
                    self.stats["class_counts"][class_name] += 1

    def process_class_directory(self, class_name: str):
        """Process all images in a class directory"""
        input_class_dir = self.input_dir / class_name
        output_class_dir = self.output_dir / class_name

        if not input_class_dir.exists():
            logger.warning(f"Input directory not found: {input_class_dir}")
            return

        logger.info(f"Processing {class_name} images...")

        # Get all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".dcm", ".dicom"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_class_dir.glob(f"*{ext}"))
            image_files.extend(input_class_dir.glob(f"*{ext.upper()}"))

        # Process each image
        for image_path in tqdm(image_files, desc=f"Processing {class_name}"):
            output_path = output_class_dir / f"{image_path.stem}.png"
            self.process_single_image(image_path, output_path, class_name)

    def process_splits(self):
        """Process train/validation/test splits"""
        splits_dir = self.input_dir / "splits"
        if not splits_dir.exists():
            logger.warning("Splits directory not found, skipping split processing")
            return

        logger.info("Processing train/validation/test splits...")

        for split in ["train", "val", "test"]:
            split_input_dir = splits_dir / split
            split_output_dir = self.output_dir / "splits" / split

            if not split_input_dir.exists():
                continue

            logger.info(f"Processing {split} split...")

            for class_name in ["normal", "benign", "malignant"]:
                class_input_dir = split_input_dir / class_name
                class_output_dir = split_output_dir / class_name

                if not class_input_dir.exists():
                    continue

                # Get all image files
                image_extensions = [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tiff",
                    ".dcm",
                    ".dicom",
                ]
                image_files = []
                for ext in image_extensions:
                    image_files.extend(class_input_dir.glob(f"*{ext}"))
                    image_files.extend(class_input_dir.glob(f"*{ext.upper()}"))

                # Process each image
                for image_path in tqdm(
                    image_files, desc=f"Processing {split}/{class_name}"
                ):
                    output_path = class_output_dir / f"{image_path.stem}.png"
                    self.process_single_image(image_path, output_path, class_name)

    def create_preprocessing_report(self):
        """Create a comprehensive preprocessing report"""
        report_path = self.output_dir / "preprocessing_report.txt"

        with open(report_path, "w") as f:
            f.write("MAMMOGRAM PREPROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Target Size: {self.target_size}\n")
            f.write(f"Normalization Range: {self.normalize_range}\n")
            f.write(f"DICOM Conversion: {self.convert_dicom}\n")
            f.write(f"Data Augmentation: {self.apply_augmentation}\n")
            f.write(f"Augmentation Factor: {self.augmentation_factor}\n\n")

            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images Processed: {self.stats['total_processed']}\n")
            f.write(f"DICOM Files Converted: {self.stats['dicom_converted']}\n")
            f.write(f"Augmented Images Created: {self.stats['augmented_images']}\n")
            f.write(f"Errors Encountered: {self.stats['errors']}\n\n")

            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for class_name, count in self.stats["class_counts"].items():
                f.write(f"{class_name.capitalize()}: {count} images\n")

        logger.info(f"Preprocessing report saved to {report_path}")

    def create_visualization(self):
        """Create visualization of preprocessed images"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Preprocessed Mammogram Images", fontsize=16)

        classes = ["normal", "benign", "malignant"]

        for i, class_name in enumerate(classes):
            class_dir = self.output_dir / class_name
            if class_dir.exists():
                # Find first image in class
                image_files = list(class_dir.glob("*.png"))
                if image_files:
                    # Load and display original
                    img = cv2.imread(str(image_files[0]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f"{class_name.capitalize()} - Original")
                    axes[0, i].axis("off")

                    # Load and display augmented (if exists)
                    aug_files = [f for f in image_files if "aug" in f.name]
                    if aug_files:
                        img_aug = cv2.imread(str(aug_files[0]))
                        img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
                        axes[1, i].imshow(img_aug)
                        axes[1, i].set_title(f"{class_name.capitalize()} - Augmented")
                        axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "preprocessing_visualization.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"Visualization saved to {self.output_dir / 'preprocessing_visualization.png'}"
        )

    def run(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting mammogram preprocessing pipeline...")

        # Process main class directories
        for class_name in ["normal", "benign", "malignant"]:
            self.process_class_directory(class_name)

        # Process splits
        self.process_splits()

        # Create report and visualization
        self.create_preprocessing_report()
        self.create_visualization()

        logger.info("Preprocessing pipeline completed!")
        logger.info(f"Total images processed: {self.stats['total_processed']}")
        logger.info(f"Augmented images created: {self.stats['augmented_images']}")
        logger.info(f"Errors: {self.stats['errors']}")


def main():
    """Main function to run the preprocessing pipeline"""
    parser = argparse.ArgumentParser(
        description="Mammogram Image Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        default="organized_dataset",
        help="Input directory containing organized images",
    )
    parser.add_argument(
        "--output-dir",
        default="preprocessed_dataset",
        help="Output directory for preprocessed images",
    )
    parser.add_argument(
        "--target-size",
        nargs=2,
        type=int,
        default=[224, 224],
        help="Target image size (width height)",
    )
    parser.add_argument(
        "--no-dicom-convert", action="store_true", help="Skip DICOM file conversion"
    )
    parser.add_argument(
        "--no-augmentation", action="store_true", help="Skip data augmentation"
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=2,
        help="Number of augmented versions per image",
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = MammogramPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        convert_dicom=not args.no_dicom_convert,
        apply_augmentation=not args.no_augmentation,
        augmentation_factor=args.augmentation_factor,
    )

    # Run preprocessing
    preprocessor.run()


if __name__ == "__main__":
    main()
