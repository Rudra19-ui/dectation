#!/usr/bin/env python3
"""
CBIS-DDSM Data Processing Script
This script processes downloaded CBIS-DDSM DICOM files into our training format
"""

import glob
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split


def process_dicom_to_image(dicom_path, output_path, size=(224, 224)):
    """Convert DICOM file to PNG image"""
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(dicom_path)

        # Get pixel data
        pixel_array = dcm.pixel_array

        # Normalize to 0-255
        if pixel_array.max() > 255:
            pixel_array = (
                (pixel_array - pixel_array.min())
                / (pixel_array.max() - pixel_array.min())
                * 255
            ).astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(pixel_array)

        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)

        # Save as PNG
        img.save(output_path, "PNG")
        return True

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
        return False


def extract_labels_from_metadata(dicom_dir):
    """Extract pathology labels from DICOM metadata"""
    labels = []

    for dicom_file in glob.glob(os.path.join(dicom_dir, "*.dcm")):
        try:
            dcm = pydicom.dcmread(dicom_file)

            # Extract pathology information
            # Note: CBIS-DDSM has specific metadata fields for pathology
            pathology = getattr(dcm, "Pathology", "Unknown")

            # Map to our classes
            if "BENIGN" in str(pathology).upper():
                label = "benign"
            elif "MALIGNANT" in str(pathology).upper():
                label = "malignant"
            else:
                label = "normal"

            labels.append(
                {
                    "filename": os.path.basename(dicom_file).replace(".dcm", ".png"),
                    "label": label,
                    "dicom_path": dicom_file,
                }
            )

        except Exception as e:
            print(f"Error reading metadata from {dicom_file}: {e}")

    return labels


def main():
    """Main processing function"""
    print("CBIS-DDSM Data Processing")
    print("=" * 50)

    # Configuration
    dicom_dir = "dataset4/cbisddsm_download"  # Your downloaded DICOM files
    output_dir = "data/images"

    if not os.path.exists(dicom_dir):
        print(f"DICOM directory not found: {dicom_dir}")
        print("Please download CBIS-DDSM dataset first")
        return

    print(f"Processing DICOM files from: {dicom_dir}")

    # Extract labels from metadata
    print("Extracting labels from DICOM metadata...")
    labels = extract_labels_from_metadata(dicom_dir)

    if not labels:
        print("No valid DICOM files found!")
        return

    print(f"Found {len(labels)} DICOM files")

    # Process images
    print("Converting DICOM to PNG images...")
    processed_count = 0

    for item in labels:
        dicom_path = item["dicom_path"]
        output_path = os.path.join(output_dir, item["filename"])

        if process_dicom_to_image(dicom_path, output_path):
            processed_count += 1

    print(f"Successfully processed {processed_count}/{len(labels)} images")

    # Create CSV files
    if processed_count > 0:
        print("Creating dataset splits...")

        # Create DataFrame
        df = pd.DataFrame(
            [
                item
                for item in labels
                if os.path.exists(os.path.join(output_dir, item["filename"]))
            ]
        )

        # Split data
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
