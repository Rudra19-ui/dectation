#!/usr/bin/env python3
"""
Sample CBIS-DDSM Processing Script
This script processes sample data for testing
"""

import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def create_sample_images(sample_dir, output_dir):
    """Create sample images from placeholder files"""
    print("Creating sample images...")

    os.makedirs(output_dir, exist_ok=True)

    sample_data = []

    for file_path in glob.glob(os.path.join(sample_dir, "*.dcm")):
        filename = os.path.basename(file_path)
        label = filename.split("_")[0]  # Extract label from filename

        # Create a sample image (white square with label text)
        img = Image.new("RGB", (224, 224), color="white")

        # Save the image
        output_filename = filename.replace(".dcm", ".png")
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)

        sample_data.append({"filename": output_filename, "label": label})

    return sample_data


def main():
    """Main processing function"""
    print("Sample CBIS-DDSM Data Processing")
    print("=" * 50)

    # Configuration
    sample_dir = "dataset4/cbisddsm_sample"
    output_dir = "data/images"

    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        print("Please run the download script first")
        return

    print(f"Processing sample files from: {sample_dir}")

    # Create sample images
    sample_data = create_sample_images(sample_dir, output_dir)

    if not sample_data:
        print("No sample files found!")
        return

    print(f"Created {len(sample_data)} sample images")

    # Create CSV files
    print("Creating dataset splits...")

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # For small datasets, use simple splitting without stratification
    if len(df) < 10:
        print("Small dataset detected, using simple split...")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    else:
        # Use stratified split for larger datasets
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df["label"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
        )

    # Save CSV files
    train_df[["filename", "label"]].to_csv("data/train.csv", index=False)
    val_df[["filename", "label"]].to_csv("data/val.csv", index=False)
    test_df[["filename", "label"]].to_csv("data/test.csv", index=False)

    print("Dataset splits created:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")

    # Show label distribution
    print("\nLabel distribution:")
    for split_name, split_df in [
        ("Train", train_df),
        ("Val", val_df),
        ("Test", test_df),
    ]:
        print(f"   {split_name}:")
        label_counts = split_df["label"].value_counts()
        for label, count in label_counts.items():
            print(f"     {label}: {count}")

    print("\nProcessing complete!")
    print("You can now run training with:")
    print("   python train_model.py")


if __name__ == "__main__":
    main()
