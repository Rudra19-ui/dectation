#!/usr/bin/env python3
"""
Dataset Splitting Script for Mammogram Classification
====================================================

This script splits the preprocessed mammogram dataset into:
- 70% Training
- 15% Validation
- 15% Testing

Ensures balanced class distribution across all splits.

Features:
- Stratified sampling to maintain class balance
- Handles both main dataset and splits
- Creates CSV files with split information
- Generates detailed statistics and visualizations
- Supports custom split ratios
"""

import argparse
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Comprehensive dataset splitter with balanced class distribution"""

    def __init__(
        self,
        input_dir: str = "preprocessed_dataset",
        output_dir: str = "split_dataset",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        preserve_splits: bool = True,
    ):
        """
        Initialize the dataset splitter

        Args:
            input_dir: Directory containing preprocessed images
            output_dir: Directory to save split datasets
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            preserve_splits: Whether to preserve existing splits
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.preserve_splits = preserve_splits

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Create output directories
        self._create_output_dirs()

        # Statistics tracking
        self.stats = {
            "total_images": 0,
            "class_counts": defaultdict(int),
            "split_counts": defaultdict(lambda: defaultdict(int)),
            "errors": 0,
        }

    def _create_output_dirs(self):
        """Create output directory structure"""
        # Main split directories
        for split in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                (self.output_dir / split / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        logger.info(f"Created output directory structure in {self.output_dir}")

    def _get_image_files(self, class_dir: Path) -> List[Path]:
        """Get all image files from a class directory"""
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def _copy_file(self, src: Path, dst: Path) -> bool:
        """Copy a file with error handling"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Error copying {src} to {dst}: {e}")
            self.stats["errors"] += 1
            return False

    def _split_class_data(
        self, class_name: str, image_files: List[Path]
    ) -> Dict[str, List[Path]]:
        """
        Split class data into train/val/test sets with stratification

        Args:
            class_name: Name of the class
            image_files: List of image files for this class

        Returns:
            Dictionary with 'train', 'val', 'test' lists of file paths
        """
        if not image_files:
            logger.warning(f"No images found for class {class_name}")
            return {"train": [], "val": [], "test": []}

        # Convert to list of strings for sklearn compatibility
        file_paths = [str(f) for f in image_files]

        # First split: separate test set
        train_val_files, test_files = train_test_split(
            file_paths,
            test_size=self.test_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        # Second split: separate validation from training
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            shuffle=True,
        )

        # Convert back to Path objects
        splits = {
            "train": [Path(f) for f in train_files],
            "val": [Path(f) for f in val_files],
            "test": [Path(f) for f in test_files],
        }

        # Log split statistics
        logger.info(
            f"{class_name}: {len(image_files)} total -> "
            f"train: {len(splits['train'])}, "
            f"val: {len(splits['val'])}, "
            f"test: {len(splits['test'])}"
        )

        return splits

    def _copy_split_files(self, class_name: str, splits: Dict[str, List[Path]]):
        """Copy files to their respective split directories"""
        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name / class_name

            logger.info(f"Copying {len(files)} {class_name} files to {split_name}...")

            for file_path in tqdm(files, desc=f"Copying {class_name} to {split_name}"):
                if self._copy_file(file_path, split_dir / file_path.name):
                    self.stats["split_counts"][split_name][class_name] += 1
                    self.stats["total_images"] += 1

    def split_main_dataset(self):
        """Split the main dataset (normal/benign/malignant directories)"""
        logger.info("Splitting main dataset...")

        for class_name in ["normal", "benign", "malignant"]:
            class_dir = self.input_dir / class_name

            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Get all image files for this class
            image_files = self._get_image_files(class_dir)
            self.stats["class_counts"][class_name] = len(image_files)

            if not image_files:
                logger.warning(f"No images found in {class_dir}")
                continue

            # Split the data
            splits = self._split_class_data(class_name, image_files)

            # Copy files to split directories
            self._copy_split_files(class_name, splits)

    def split_existing_splits(self):
        """Split the existing splits (if they exist)"""
        splits_dir = self.input_dir / "splits"

        if not splits_dir.exists():
            logger.info("No existing splits found, skipping split processing")
            return

        logger.info("Processing existing splits...")

        for split_name in ["train", "val", "test"]:
            split_input_dir = splits_dir / split_name

            if not split_input_dir.exists():
                continue

            logger.info(f"Processing {split_name} split...")

            for class_name in ["normal", "benign", "malignant"]:
                class_dir = split_input_dir / class_name

                if not class_dir.exists():
                    continue

                # Get all image files for this class
                image_files = self._get_image_files(class_dir)

                if not image_files:
                    continue

                # Copy all files to the corresponding split directory
                split_output_dir = self.output_dir / split_name / class_name

                logger.info(
                    f"Copying {len(image_files)} {class_name} files from {split_name}..."
                )

                for file_path in tqdm(
                    image_files, desc=f"Copying {split_name}/{class_name}"
                ):
                    if self._copy_file(file_path, split_output_dir / file_path.name):
                        self.stats["split_counts"][split_name][class_name] += 1
                        self.stats["total_images"] += 1

    def create_split_csv(self):
        """Create CSV files with split information"""
        logger.info("Creating split CSV files...")

        # Create main split CSV
        split_data = []

        for split_name in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                split_dir = self.output_dir / split_name / class_name

                if split_dir.exists():
                    image_files = list(split_dir.glob("*.png"))

                    for file_path in image_files:
                        split_data.append(
                            {
                                "filename": file_path.name,
                                "class": class_name,
                                "split": split_name,
                                "path": str(file_path.relative_to(self.output_dir)),
                            }
                        )

        # Create DataFrame and save
        df = pd.DataFrame(split_data)
        csv_path = self.output_dir / "dataset_splits.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Split CSV saved to {csv_path}")

        # Create summary CSV
        summary_data = []
        for split_name in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                count = self.stats["split_counts"][split_name][class_name]
                summary_data.append(
                    {"split": split_name, "class": class_name, "count": count}
                )

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / "split_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        logger.info(f"Summary CSV saved to {summary_csv_path}")

    def create_split_report(self):
        """Create a comprehensive split report"""
        report_path = self.output_dir / "split_report.txt"

        with open(report_path, "w") as f:
            f.write("DATASET SPLIT REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Train Ratio: {self.train_ratio:.1%}\n")
            f.write(f"Validation Ratio: {self.val_ratio:.1%}\n")
            f.write(f"Test Ratio: {self.test_ratio:.1%}\n")
            f.write(f"Random State: {self.random_state}\n\n")

            f.write("SPLIT STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images: {self.stats['total_images']}\n")
            f.write(f"Errors: {self.stats['errors']}\n\n")

            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for class_name, count in self.stats["class_counts"].items():
                f.write(f"{class_name.capitalize()}: {count} images\n")
            f.write("\n")

            f.write("SPLIT BREAKDOWN:\n")
            f.write("-" * 20 + "\n")
            for split_name in ["train", "val", "test"]:
                f.write(f"\n{split_name.upper()} SET:\n")
                total_split = 0
                for class_name in ["normal", "benign", "malignant"]:
                    count = self.stats["split_counts"][split_name][class_name]
                    total_split += count
                    f.write(f"  {class_name.capitalize()}: {count} images\n")
                f.write(f"  Total: {total_split} images\n")

        logger.info(f"Split report saved to {report_path}")

    def create_visualization(self):
        """Create visualizations of the split distribution"""
        # Create split distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Split distribution by class
        split_data = []
        for split_name in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                count = self.stats["split_counts"][split_name][class_name]
                split_data.append(
                    {
                        "split": split_name.capitalize(),
                        "class": class_name.capitalize(),
                        "count": count,
                    }
                )

        df_plot = pd.DataFrame(split_data)

        # Create stacked bar plot
        pivot_df = df_plot.pivot(index="split", columns="class", values="count")
        pivot_df.plot(
            kind="bar",
            stacked=True,
            ax=axes[0],
            color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
        )
        axes[0].set_title("Dataset Split Distribution by Class")
        axes[0].set_ylabel("Number of Images")
        axes[0].legend(title="Class")
        axes[0].tick_params(axis="x", rotation=0)

        # Plot 2: Class distribution across splits
        sns.heatmap(pivot_df, annot=True, fmt="d", cmap="Blues", ax=axes[1])
        axes[1].set_title("Class Distribution Heatmap")
        axes[1].set_ylabel("Split")
        axes[1].set_xlabel("Class")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "split_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Create pie chart for overall class distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        class_totals = defaultdict(int)
        for split_name in ["train", "val", "test"]:
            for class_name in ["normal", "benign", "malignant"]:
                class_totals[class_name] += self.stats["split_counts"][split_name][
                    class_name
                ]

        labels = [name.capitalize() for name in class_totals.keys()]
        sizes = list(class_totals.values())
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )
        ax.set_title("Overall Class Distribution")

        plt.savefig(
            self.output_dir / "class_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Visualizations saved to {self.output_dir}")

    def validate_splits(self):
        """Validate that splits are properly balanced"""
        logger.info("Validating splits...")

        issues = []

        for split_name in ["train", "val", "test"]:
            split_counts = []
            for class_name in ["normal", "benign", "malignant"]:
                count = self.stats["split_counts"][split_name][class_name]
                split_counts.append(count)

            # Check if any class has 0 images
            if any(count == 0 for count in split_counts):
                issues.append(f"{split_name} split has 0 images for some classes")

            # Check for extreme imbalance (more than 10x difference)
            if max(split_counts) > 0 and min(split_counts) > 0:
                ratio = max(split_counts) / min(split_counts)
                if ratio > 10:
                    issues.append(
                        f"{split_name} split has extreme class imbalance (ratio: {ratio:.1f})"
                    )

        if issues:
            logger.warning("Split validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("All splits validated successfully!")

        return len(issues) == 0

    def run(self):
        """Run the complete dataset splitting pipeline"""
        logger.info("Starting dataset splitting pipeline...")

        # Split main dataset
        self.split_main_dataset()

        # Split existing splits if requested
        if self.preserve_splits:
            self.split_existing_splits()

        # Create CSV files
        self.create_split_csv()

        # Create report and visualizations
        self.create_split_report()
        self.create_visualization()

        # Validate splits
        self.validate_splits()

        logger.info("Dataset splitting pipeline completed!")
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Errors: {self.stats['errors']}")


def main():
    """Main function to run the dataset splitting pipeline"""
    parser = argparse.ArgumentParser(description="Dataset Splitting Pipeline")
    parser.add_argument(
        "--input-dir",
        default="preprocessed_dataset",
        help="Input directory containing preprocessed images",
    )
    parser.add_argument(
        "--output-dir",
        default="split_dataset",
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Proportion for training set"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Proportion for validation set"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Proportion for test set"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-preserve-splits",
        action="store_true",
        help="Don't preserve existing splits",
    )

    args = parser.parse_args()

    # Initialize splitter
    splitter = DatasetSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        preserve_splits=not args.no_preserve_splits,
    )

    # Run splitting
    splitter.run()


if __name__ == "__main__":
    main()
