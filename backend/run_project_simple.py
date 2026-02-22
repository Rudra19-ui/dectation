#!/usr/bin/env python3
"""
Simplified Breast Cancer Classification Project Runner
Avoids multiprocessing issues and runs step by step
"""

import os
import sys
import warnings

import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set environment variables to avoid multiprocessing issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")

    required_packages = [
        "tensorflow",
        "numpy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "pandas",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "sklearn":
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - NOT FOUND")

    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies found!")
    return True


def check_dataset():
    """Check if dataset exists and has correct structure"""
    print("\n🔍 Checking dataset...")

    dataset_path = r"E:\rudra\project\dataset"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False

    print(f"✅ Dataset path found: {dataset_path}")

    # Check for class folders
    class_folders = ["benign", "malignant", "normal"]
    total_images = 0

    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"❌ Missing class folder: {class_name}")
            return False

        # Count images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            import glob

            image_files.extend(glob.glob(os.path.join(class_path, ext)))

        count = len(image_files)
        total_images += count
        print(f"✅ {class_name}: {count} images")

    if total_images == 0:
        print("❌ No images found in dataset")
        return False

    print(f"✅ Total images: {total_images}")
    return True


def run_master_script():
    """Run the master script with error handling"""
    print("\n🚀 Running master breast cancer classifier...")

    try:
        # Import and run the master script
        from master_breast_cancer_classifier import BreastCancerMasterClassifier

        # Create classifier
        classifier = BreastCancerMasterClassifier()

        # Run complete pipeline
        success = classifier.run_complete_pipeline(
            num_samples=3
        )  # Reduced samples for testing

        if success:
            print("\n🎉 Project completed successfully!")
            return True
        else:
            print("\n❌ Project failed!")
            return False

    except Exception as e:
        print(f"\n❌ Error running master script: {e}")
        print("Full error details:")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("=" * 80)
    print("🏥 Breast Cancer Classification Project Runner")
    print("=" * 80)

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return False

    # Step 2: Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed. Please ensure dataset is properly organized.")
        return False

    # Step 3: Run master script
    success = run_master_script()

    if success:
        print("\n" + "=" * 80)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📁 Check the generated files for results:")
        print("   - breast_cancer_master_model.h5 (trained model)")
        print("   - predictions_*.csv (detailed predictions)")
        print("   - confusion_matrix_*.png (confusion matrix plot)")
        print("   - classification_report_*.txt (detailed report)")
        print("   - summary_*.txt (summary statistics)")
    else:
        print("\n" + "=" * 80)
        print("❌ PROJECT FAILED!")
        print("=" * 80)
        print("Please check the error messages above and try again.")

    return success


if __name__ == "__main__":
    main()
