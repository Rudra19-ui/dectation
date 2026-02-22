#!/usr/bin/env python3
"""
Dataset Verification Script
Scans and verifies the dataset folder structure for breast cancer classification
"""

import glob
import os
from pathlib import Path

import pandas as pd


def scan_dataset_folder(dataset_path="dataset"):
    """
    Scan the dataset folder and verify structure
    """
    print("=" * 60)
    print("🔍 DATASET VERIFICATION REPORT")
    print("=" * 60)

    # Check if dataset folder exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        print("\n📁 Available folders in current directory:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  📂 {item}/")
        return False

    print(f"✅ Dataset folder '{dataset_path}' found")

    # Expected subfolders
    expected_folders = ["benign", "malignant", "normal"]
    found_folders = []
    missing_folders = []

    # Scan for subfolders
    for folder in expected_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            found_folders.append(folder)
            print(f"✅ Found subfolder: {folder}/")
        else:
            missing_folders.append(folder)
            print(f"❌ Missing subfolder: {folder}/")

    print(f"\n📊 Folder Summary:")
    print(f"  Found: {len(found_folders)}/{len(expected_folders)} folders")
    print(f"  Missing: {len(missing_folders)} folders")

    if missing_folders:
        print(f"\n❌ Missing folders: {', '.join(missing_folders)}")
        return False

    # Scan files in each folder
    print(f"\n📁 File Analysis:")
    total_files = 0
    valid_files = 0
    invalid_files = []

    for folder in found_folders:
        folder_path = os.path.join(dataset_path, folder)
        files = os.listdir(folder_path)

        print(f"\n📂 {folder}/:")
        folder_files = 0
        folder_valid = 0

        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                folder_files += 1
                total_files += 1

                # Check file extension
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    folder_valid += 1
                    valid_files += 1
                    print(f"  ✅ {file}")
                else:
                    invalid_files.append((folder, file))
                    print(f"  ❌ {file} (invalid format)")

        print(f"  📊 {folder_valid}/{folder_files} valid files in {folder}/")

    print(f"\n📊 Overall Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Valid files (.jpg/.png): {valid_files}")
    print(f"  Invalid files: {len(invalid_files)}")

    if invalid_files:
        print(f"\n❌ Invalid files found:")
        for folder, file in invalid_files:
            print(f"  {folder}/{file}")

    # Check for minimum file requirements
    min_files_per_class = 10
    print(f"\n🔍 Class Balance Check (minimum {min_files_per_class} files per class):")

    for folder in found_folders:
        folder_path = os.path.join(dataset_path, folder)
        valid_count = len(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if valid_count >= min_files_per_class:
            print(f"  ✅ {folder}: {valid_count} files")
        else:
            print(f"  ⚠️ {folder}: {valid_count} files (below minimum)")

    return len(missing_folders) == 0 and len(invalid_files) == 0


def create_dataset_structure():
    """
    Create the proper dataset structure if it doesn't exist
    """
    print("\n" + "=" * 60)
    print("🔧 CREATING DATASET STRUCTURE")
    print("=" * 60)

    dataset_path = "dataset"
    expected_folders = ["benign", "malignant", "normal"]

    # Create main dataset folder
    os.makedirs(dataset_path, exist_ok=True)
    print(f"✅ Created dataset folder: {dataset_path}/")

    # Create subfolders
    for folder in expected_folders:
        folder_path = os.path.join(dataset_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✅ Created subfolder: {folder}/")

    print(f"\n📁 Dataset structure created:")
    print(f"  {dataset_path}/")
    for folder in expected_folders:
        print(f"  ├── {folder}/")
    print(f"  └── (ready for images)")

    return True


def suggest_fixes():
    """
    Suggest fixes for common issues
    """
    print("\n" + "=" * 60)
    print("💡 SUGGESTIONS")
    print("=" * 60)

    print("1. 📁 If dataset folder doesn't exist:")
    print("   - Run: python verify_dataset.py --create")
    print(
        "   - Or manually create: dataset/benign/, dataset/malignant/, dataset/normal/"
    )

    print("\n2. 🖼️ If images are in wrong format:")
    print("   - Convert .bmp, .tiff, .dcm files to .jpg or .png")
    print("   - Use image editing software or Python PIL library")

    print("\n3. 📂 If images are in wrong folders:")
    print("   - Move images to correct class folders")
    print("   - Ensure naming convention is clear")

    print("\n4. 🔍 If files are missing:")
    print("   - Check if images were moved during preprocessing")
    print("   - Look in: data/images/, organized_dataset/, split_dataset/")

    print("\n5. 📊 For better results:")
    print("   - Aim for at least 50 images per class")
    print("   - Use consistent image sizes (224x224 recommended)")
    print("   - Ensure good quality images")


def main():
    """
    Main function to run dataset verification
    """
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        create_dataset_structure()
        return

    # Scan existing dataset
    success = scan_dataset_folder()

    if not success:
        suggest_fixes()
        print(f"\n❌ Dataset verification failed!")
        print(f"Run 'python verify_dataset.py --create' to create proper structure")
    else:
        print(f"\n✅ Dataset verification passed!")
        print(f"Your dataset is ready for training!")


if __name__ == "__main__":
    main()
