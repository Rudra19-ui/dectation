#!/usr/bin/env python3
"""
Process Thermal/Ultrasound Dataset for Breast Cancer Classification
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def process_thermal_dataset():
    """Process the thermal/ultrasound dataset"""
    print("🔬 Processing Thermal/Ultrasound Dataset")
    print("=" * 50)

    # Source and destination paths
    source_path = Path("dataset4/archive/Dataset_BUSI_with_GT")
    dest_path = Path("data/thermal_images")

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    # Classes to process
    classes = ["benign", "malignant", "normal"]

    # Statistics
    stats = {
        "total_images": 0,
        "processed_images": 0,
        "skipped_images": 0,
        "class_counts": {},
    }

    # Process each class
    for class_name in classes:
        class_source = source_path / class_name
        class_dest = dest_path / class_name

        if not class_source.exists():
            print(f"⚠️ Class directory not found: {class_source}")
            continue

        # Create destination directory for this class
        class_dest.mkdir(exist_ok=True)

        # Get all PNG files (excluding mask files)
        image_files = [
            f
            for f in class_source.glob("*.png")
            if not f.name.endswith("_mask.png") and not f.name.endswith("_mask_1.png")
        ]

        print(f"\n📁 Processing {class_name}: {len(image_files)} images")
        stats["class_counts"][class_name] = 0

        for i, image_file in enumerate(image_files):
            try:
                # Load and process image
                image = Image.open(image_file)

                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Resize to 224x224
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

                # Save processed image
                new_filename = f"{class_name}_{i+1:03d}.png"
                output_path = class_dest / new_filename
                image.save(output_path, "PNG")

                stats["class_counts"][class_name] += 1
                stats["processed_images"] += 1

                if (i + 1) % 10 == 0:
                    print(f"  ✅ Processed {i + 1}/{len(image_files)} images")

            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
                stats["skipped_images"] += 1

        stats["total_images"] += len(image_files)

    # Create CSV files for training
    create_thermal_csv_files(dest_path, stats)

    print(f"\n📊 Processing Complete!")
    print(f"   Total images found: {stats['total_images']}")
    print(f"   Successfully processed: {stats['processed_images']}")
    print(f"   Skipped: {stats['skipped_images']}")
    print(f"   Class distribution:")
    for class_name, count in stats["class_counts"].items():
        print(f"     {class_name}: {count} images")

    return stats


def create_thermal_csv_files(dest_path, stats):
    """Create CSV files for the thermal dataset"""
    print("\n📝 Creating CSV files...")

    # Collect all image paths and labels
    data = []
    for class_name in ["benign", "malignant", "normal"]:
        class_dir = dest_path / class_name
        if class_dir.exists():
            for image_file in class_dir.glob("*.png"):
                data.append(
                    {
                        "image_path": str(image_file),
                        "label": class_name,
                        "dataset": "thermal",
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Split into train/val/test (70/15/15)
    from sklearn.model_selection import train_test_split

    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label"]
    )

    # Second split: 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    # Save CSV files
    train_df.to_csv("data/thermal_train.csv", index=False)
    val_df.to_csv("data/thermal_val.csv", index=False)
    test_df.to_csv("data/thermal_test.csv", index=False)

    print(f"✅ CSV files created:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")

    # Save combined dataset info
    dataset_info = {
        "dataset_type": "thermal_ultrasound",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": df["label"].value_counts().to_dict(),
        "processed_date": datetime.now().isoformat(),
    }

    # Save dataset info
    import json

    with open("data/thermal_dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"✅ Dataset info saved to: data/thermal_dataset_info.json")


if __name__ == "__main__":
    process_thermal_dataset()
