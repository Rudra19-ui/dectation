#!/usr/bin/env python3
"""
BUSI Dataset Filter
Filters BUSI dataset to keep only original images, removing mask and ground truth files
"""

import argparse
import os
import shutil
from pathlib import Path


def filter_busi_dataset(input_dir, output_dir):
    """
    Filter BUSI dataset to keep only original mammogram images

    Args:
        input_dir: Path to BUSI dataset
        output_dir: Path to output filtered dataset
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    for class_name in ["benign", "malignant", "normal"]:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)

    stats = {
        "benign": 0,
        "malignant": 0,
        "normal": 0,
        "total_processed": 0,
        "filtered_out": 0,
    }

    print("Filtering BUSI dataset...")

    for class_name in ["benign", "malignant", "normal"]:
        class_input_dir = input_path / class_name
        class_output_dir = output_path / class_name

        if not class_input_dir.exists():
            print(f"⚠️ Warning: {class_input_dir} not found")
            continue

        print(f"\nProcessing {class_name} class...")

        for file_path in class_input_dir.iterdir():
            if file_path.is_file():
                stats["total_processed"] += 1
                filename = file_path.name.lower()

                # Skip mask files and ground truth files
                if any(
                    skip_pattern in filename
                    for skip_pattern in ["mask", "_mask", "gt", "_gt", "ground_truth"]
                ):
                    stats["filtered_out"] += 1
                    print(f"   Filtered out: {file_path.name}")
                    continue

                # Keep only image files
                if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    # Generate new filename to avoid conflicts
                    new_filename = (
                        f"busi_{class_name}_{stats[class_name]:04d}{file_path.suffix}"
                    )
                    output_file_path = class_output_dir / new_filename

                    shutil.copy2(file_path, output_file_path)
                    stats[class_name] += 1

                    if stats[class_name] % 100 == 0:
                        print(
                            f"   Processed {stats[class_name]} {class_name} images..."
                        )

    # Print summary
    print(f"\nBUSI Dataset Filtering Complete!")
    print(f"   Total files processed: {stats['total_processed']}")
    print(f"   Files filtered out: {stats['filtered_out']}")
    print(f"   Files kept:")
    for class_name in ["benign", "malignant", "normal"]:
        print(f"     {class_name.capitalize()}: {stats[class_name]} images")
    print(
        f"   Total images kept: {sum(stats[c] for c in ['benign', 'malignant', 'normal'])}"
    )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BUSI dataset")
    parser.add_argument(
        "--input_dir",
        default="dataset_integration/step2_busi_copied",
        help="Path to BUSI dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset_integration/step2_busi_filtered",
        help="Path to output filtered dataset",
    )

    args = parser.parse_args()

    filter_busi_dataset(args.input_dir, args.output_dir)
