#!/usr/bin/env python3
"""
Dataset Organization Script for Mammogram Classification
Organizes mammogram images into three categories: normal, benign, malignant

This script handles:
- Multiple data sources (enhanced dataset, CBIS-DDSM, etc.)
- Different file formats (PNG, JPEG, DICOM)
- Various labeling schemes (CSV files, filename patterns)
- Automatic folder creation and file organization
"""

import argparse
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    def __init__(self, output_dir: str = "organized_dataset"):
        """
        Initialize the dataset organizer

        Args:
            output_dir: Directory to store organized images
        """
        self.output_dir = Path(output_dir)
        self.categories = ["normal", "benign", "malignant"]

        # Create output directories
        for category in self.categories:
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {category: 0 for category in self.categories}
        self.errors = []

    def organize_enhanced_dataset(self, csv_file: str, images_dir: str):
        """
        Organize the enhanced dataset using CSV labels

        Args:
            csv_file: Path to CSV file with labels
            images_dir: Directory containing images
        """
        logger.info(f"Organizing enhanced dataset from {csv_file}")

        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Found {len(df)} entries in CSV")

            for idx, row in df.iterrows():
                filename = row["filename"]
                label = row["label"].lower()

                if label not in self.categories:
                    logger.warning(f"Unknown label '{label}' for file {filename}")
                    continue

                source_path = Path(images_dir) / filename
                if source_path.exists():
                    dest_path = self.output_dir / label / filename
                    shutil.copy2(source_path, dest_path)
                    self.stats[label] += 1
                    logger.debug(f"Copied {filename} to {label}/")
                else:
                    error_msg = f"File not found: {source_path}"
                    self.errors.append(error_msg)
                    logger.warning(error_msg)

        except Exception as e:
            logger.error(f"Error processing enhanced dataset: {e}")
            self.errors.append(f"Enhanced dataset error: {e}")

    def organize_cbisddsm_dataset(self, csv_file: str, base_dir: str):
        """
        Organize CBIS-DDSM dataset using pathology information

        Args:
            csv_file: Path to CBIS-DDSM CSV file
            base_dir: Base directory containing CBIS-DDSM images
        """
        logger.info(f"Organizing CBIS-DDSM dataset from {csv_file}")

        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Found {len(df)} entries in CBIS-DDSM CSV")

            for idx, row in df.iterrows():
                pathology = row["pathology"].upper()
                image_path = row["cropped image file path"]

                # Map pathology to categories
                if pathology == "MALIGNANT":
                    category = "malignant"
                elif pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
                    category = "benign"
                else:
                    logger.warning(f"Unknown pathology '{pathology}' for {image_path}")
                    continue

                # Construct full path
                full_path = Path(base_dir) / image_path

                if full_path.exists():
                    # Generate unique filename
                    filename = f"cbisddsm_{category}_{idx:04d}.dcm"
                    dest_path = self.output_dir / category / filename

                    shutil.copy2(full_path, dest_path)
                    self.stats[category] += 1
                    logger.debug(f"Copied {image_path} to {category}/{filename}")
                else:
                    error_msg = f"File not found: {full_path}"
                    self.errors.append(error_msg)
                    logger.warning(error_msg)

        except Exception as e:
            logger.error(f"Error processing CBIS-DDSM dataset: {e}")
            self.errors.append(f"CBIS-DDSM dataset error: {e}")

    def organize_by_filename_pattern(self, images_dir: str):
        """
        Organize images based on filename patterns

        Args:
            images_dir: Directory containing images
        """
        logger.info(f"Organizing images by filename pattern from {images_dir}")

        images_path = Path(images_dir)
        if not images_path.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return

        # Supported image extensions
        image_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}

        for file_path in images_path.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                filename = file_path.name.lower()

                # Determine category from filename
                category = None
                if "normal" in filename:
                    category = "normal"
                elif "benign" in filename:
                    category = "benign"
                elif "malignant" in filename:
                    category = "malignant"

                if category:
                    dest_path = self.output_dir / category / file_path.name
                    shutil.copy2(file_path, dest_path)
                    self.stats[category] += 1
                    logger.debug(f"Copied {file_path.name} to {category}/")
                else:
                    logger.warning(f"Could not determine category for {file_path.name}")

    def convert_dicom_to_png(self, dicom_dir: str, output_dir: str = None):
        """
        Convert DICOM files to PNG format for easier handling

        Args:
            dicom_dir: Directory containing DICOM files
            output_dir: Output directory for PNG files (optional)
        """
        logger.info(f"Converting DICOM files from {dicom_dir}")

        if output_dir is None:
            output_dir = self.output_dir / "converted_png"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dicom_path = Path(dicom_dir)
        converted_count = 0

        for dicom_file in dicom_path.rglob("*.dcm"):
            try:
                # Read DICOM file
                ds = pydicom.dcmread(str(dicom_file))

                # Convert to PIL Image
                if hasattr(ds, "pixel_array"):
                    image = Image.fromarray(ds.pixel_array)

                    # Normalize if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Save as PNG
                    png_filename = dicom_file.stem + ".png"
                    png_path = output_path / png_filename
                    image.save(png_path)

                    converted_count += 1
                    logger.debug(f"Converted {dicom_file.name} to {png_filename}")

            except Exception as e:
                error_msg = f"Error converting {dicom_file}: {e}"
                self.errors.append(error_msg)
                logger.warning(error_msg)

        logger.info(f"Converted {converted_count} DICOM files to PNG")
        return str(output_path)

    def create_dataset_summary(self):
        """Create a summary of the organized dataset"""
        logger.info("Creating dataset summary...")

        summary = {
            "total_images": sum(self.stats.values()),
            "categories": self.stats.copy(),
            "errors": len(self.errors),
        }

        # Create summary file
        summary_file = self.output_dir / "dataset_summary.txt"
        with open(summary_file, "w") as f:
            f.write("DATASET ORGANIZATION SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Images: {summary['total_images']}\n\n")

            for category, count in self.stats.items():
                f.write(f"{category.capitalize()}: {count} images\n")

            f.write(f"\nErrors: {summary['errors']}\n")

            if self.errors:
                f.write("\nERROR LOG:\n")
                for error in self.errors[:10]:  # Show first 10 errors
                    f.write(f"- {error}\n")
                if len(self.errors) > 10:
                    f.write(f"... and {len(self.errors) - 10} more errors\n")

        # Print summary to console
        logger.info("Dataset Summary:")
        logger.info(f"Total Images: {summary['total_images']}")
        for category, count in self.stats.items():
            logger.info(f"{category.capitalize()}: {count} images")
        logger.info(f"Errors: {summary['errors']}")

        return summary

    def create_train_val_test_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Create train/validation/test splits for each category

        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        logger.info("Creating train/validation/test splits...")

        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for category in self.categories:
            category_dir = self.output_dir / category
            if not category_dir.exists():
                continue

            # Get all images in category
            images = list(category_dir.glob("*"))
            np.random.shuffle(images)

            # Calculate split indices
            n_images = len(images)
            train_end = int(n_images * train_ratio)
            val_end = train_end + int(n_images * val_ratio)

            # Split images
            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]

            # Create split directories
            for split_name, split_images in [
                ("train", train_images),
                ("val", val_images),
                ("test", test_images),
            ]:
                split_dir = splits_dir / split_name / category
                split_dir.mkdir(parents=True, exist_ok=True)

                for img_path in split_images:
                    shutil.copy2(img_path, split_dir / img_path.name)

            logger.info(
                f"{category}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
            )

        # Create split summary
        split_summary = splits_dir / "split_summary.txt"
        with open(split_summary, "w") as f:
            f.write("TRAIN/VAL/TEST SPLIT SUMMARY\n")
            f.write("=" * 30 + "\n\n")

            for split_name in ["train", "val", "test"]:
                f.write(f"{split_name.upper()} SET:\n")
                split_dir = splits_dir / split_name
                for category in self.categories:
                    category_dir = split_dir / category
                    if category_dir.exists():
                        count = len(list(category_dir.glob("*")))
                        f.write(f"  {category}: {count} images\n")
                f.write("\n")


def main():
    """Main function to organize the dataset"""
    parser = argparse.ArgumentParser(
        description="Organize mammogram dataset into categories"
    )
    parser.add_argument(
        "--output-dir",
        default="organized_dataset",
        help="Output directory for organized images",
    )
    parser.add_argument(
        "--enhanced-csv",
        default="data/train_enhanced.csv",
        help="Path to enhanced dataset CSV",
    )
    parser.add_argument(
        "--enhanced-images",
        default="data/images",
        help="Path to enhanced dataset images",
    )
    parser.add_argument(
        "--cbisddsm-csv",
        default="dataset4/mass_case_description_train_set.csv",
        help="Path to CBIS-DDSM CSV file",
    )
    parser.add_argument(
        "--cbisddsm-base",
        default="dataset4",
        help="Base directory for CBIS-DDSM images",
    )
    parser.add_argument(
        "--convert-dicom", action="store_true", help="Convert DICOM files to PNG"
    )
    parser.add_argument(
        "--create-splits", action="store_true", help="Create train/val/test splits"
    )

    args = parser.parse_args()

    # Initialize organizer
    organizer = DatasetOrganizer(args.output_dir)

    # Organize enhanced dataset
    if os.path.exists(args.enhanced_csv):
        organizer.organize_enhanced_dataset(args.enhanced_csv, args.enhanced_images)
    else:
        logger.warning(f"Enhanced CSV not found: {args.enhanced_csv}")

    # Organize CBIS-DDSM dataset
    if os.path.exists(args.cbisddsm_csv):
        organizer.organize_cbisddsm_dataset(args.cbisddsm_csv, args.cbisddsm_base)
    else:
        logger.warning(f"CBIS-DDSM CSV not found: {args.cbisddsm_csv}")

    # Organize by filename pattern (fallback)
    if os.path.exists(args.enhanced_images):
        organizer.organize_by_filename_pattern(args.enhanced_images)

    # Convert DICOM files if requested
    if args.convert_dicom:
        png_dir = organizer.convert_dicom_to_png(args.cbisddsm_base)
        logger.info(f"DICOM files converted to: {png_dir}")

    # Create splits if requested
    if args.create_splits:
        organizer.create_train_val_test_splits()

    # Create summary
    summary = organizer.create_dataset_summary()

    logger.info("Dataset organization completed!")
    logger.info(f"Organized images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
