#!/usr/bin/env python3
"""
Breast Cancer Classification Setup Script
Creates virtual environment and installs required packages
"""

import os
import platform
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("🏥 Breast Cancer Classification Setup")
    print("=" * 60)
    print()

    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print()

    # Create virtual environment
    venv_name = "breast_cancer_env"

    if not run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
        print(
            "❌ Failed to create virtual environment. Please ensure Python is installed."
        )
        return False

    # Determine activation script based on platform
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_name, "Scripts", "activate.bat")
        python_path = os.path.join(venv_name, "Scripts", "python.exe")
        pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
    else:
        activate_script = os.path.join(venv_name, "bin", "activate")
        python_path = os.path.join(venv_name, "bin", "python")
        pip_path = os.path.join(venv_name, "bin", "pip")

    # Upgrade pip
    if not run_command(
        f'"{python_path}" -m pip install --upgrade pip', "Upgrading pip"
    ):
        print("⚠️ Failed to upgrade pip, continuing...")

    # Install required packages
    packages = [
        "tensorflow",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pillow",
        "seaborn",
        "pydicom",
    ]

    print("\n📦 Installing required packages...")
    for package in packages:
        if not run_command(f'"{pip_path}" install {package}', f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

    # Test installation
    print("\n🧪 Testing installation...")
    test_imports = [
        ("tensorflow", "import tensorflow as tf; print('TensorFlow:', tf.__version__)"),
        ("opencv", "import cv2; print('OpenCV:', cv2.__version__)"),
        ("scikit-learn", "import sklearn; print('scikit-learn:', sklearn.__version__)"),
        (
            "matplotlib",
            "import matplotlib; print('matplotlib:', matplotlib.__version__)",
        ),
        ("numpy", "import numpy; print('NumPy:', numpy.__version__)"),
    ]

    for name, import_code in test_imports:
        try:
            result = subprocess.run(
                f'"{python_path}" -c "{import_code}"',
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: Failed to import")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()

    # Print activation instructions
    if platform.system() == "Windows":
        print("To activate the environment:")
        print(f"  {venv_name}\\Scripts\\activate.bat")
        print()
        print("Or in PowerShell:")
        print(f"  .\\{venv_name}\\Scripts\\Activate.ps1")
    else:
        print("To activate the environment:")
        print(f"  source {venv_name}/bin/activate")

    print()
    print("To deactivate:")
    print("  deactivate")
    print()
    print("To run the breast cancer classifier:")
    print("  python simple_classifier.py")
    print()

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Setup completed successfully!")
    else:
        print("❌ Setup failed. Please check the error messages above.")

    input("\nPress Enter to exit...")
