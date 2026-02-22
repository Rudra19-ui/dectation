#!/usr/bin/env python3
"""
Basic Project Setup Checker
Checks environment and dataset without importing TensorFlow
"""

import glob
import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    print(f"   Python version: {sys.version}")

    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return False

    print("✅ Python version OK")
    return True


def check_project_structure():
    """Check project file structure"""
    print("\n📁 Checking project structure...")

    current_dir = Path.cwd()
    print(f"   Current directory: {current_dir}")

    # Check for key files
    key_files = [
        "master_breast_cancer_classifier.py",
        "run_project_simple.py",
        "check_project_setup.py",
    ]

    for file in key_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - NOT FOUND")

    return True


def check_dataset_structure():
    """Check dataset structure"""
    print("\n📊 Checking dataset structure...")

    dataset_path = Path(r"E:\rudra\project\dataset")

    if not dataset_path.exists():
        print(f"   ❌ Dataset path not found: {dataset_path}")
        print("   Please ensure the dataset folder exists at the specified path")
        return False

    print(f"   ✅ Dataset path found: {dataset_path}")

    # Check for class folders
    class_folders = ["benign", "malignant", "normal"]
    total_images = 0

    for class_name in class_folders:
        class_path = dataset_path / class_name

        if not class_path.exists():
            print(f"   ❌ Missing class folder: {class_name}")
            return False

        # Count images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(class_path.glob(ext))

        count = len(image_files)
        total_images += count
        print(f"   ✅ {class_name}: {count} images")

        # Show first few image names
        if count > 0:
            sample_files = [f.name for f in image_files[:3]]
            print(f"      Sample files: {sample_files}")

    if total_images == 0:
        print("   ❌ No images found in dataset")
        return False

    print(f"   ✅ Total images: {total_images}")
    return True


def check_basic_imports():
    """Check basic Python imports"""
    print("\n📦 Checking basic imports...")

    basic_packages = ["numpy", "pandas", "matplotlib"]

    for package in basic_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT FOUND")

    return True


def check_tensorflow_availability():
    """Check if TensorFlow is available"""
    print("\n🤖 Checking TensorFlow availability...")

    try:
        import tensorflow as tf

        print(f"   ✅ TensorFlow version: {tf.__version__}")

        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            print(f"   ✅ GPU detected: {len(gpus)} GPU(s)")
        else:
            print("   ℹ️ No GPU detected, will use CPU")

        return True

    except ImportError:
        print("   ❌ TensorFlow not installed")
        print("   Please install TensorFlow: pip install tensorflow")
        return False
    except Exception as e:
        print(f"   ⚠️ TensorFlow error: {e}")
        return False


def main():
    """Main function"""
    print("=" * 80)
    print("🔍 Breast Cancer Classification - Project Setup Checker")
    print("=" * 80)

    checks = []

    # Run all checks
    checks.append(check_python_version())
    checks.append(check_project_structure())
    checks.append(check_dataset_structure())
    checks.append(check_basic_imports())
    checks.append(check_tensorflow_availability())

    # Summary
    print("\n" + "=" * 80)
    print("📊 SETUP CHECK SUMMARY")
    print("=" * 80)

    passed_checks = sum(checks)
    total_checks = len(checks)

    print(f"✅ Passed checks: {passed_checks}/{total_checks}")

    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Project is ready to run.")
        print("\n🚀 You can now run the project with:")
        print("   python master_breast_cancer_classifier.py")
        print("   or")
        print("   python run_project_simple.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print(
            "1. Install missing packages: pip install tensorflow numpy pandas matplotlib"
        )
        print("2. Ensure dataset is properly organized at E:\\rudra\\project\\dataset")
        print("3. Check that all required files are present")

    print("\n" + "=" * 80)

    return passed_checks == total_checks


if __name__ == "__main__":
    main()
