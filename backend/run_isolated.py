#!/usr/bin/env python3
"""
Isolated Runner for Breast Cancer Classification
Runs the master classifier in isolation to avoid multiprocessing issues
"""

import multiprocessing
import os
import subprocess
import sys


def run_isolated():
    """Run the master classifier in an isolated environment"""

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["PYTHONPATH"] = "."

    print("🔧 Setting up isolated environment...")
    print(f"   OMP_NUM_THREADS: {env['OMP_NUM_THREADS']}")
    print(f"   MKL_NUM_THREADS: {env['MKL_NUM_THREADS']}")
    print(f"   TF_CPP_MIN_LOG_LEVEL: {env['TF_CPP_MIN_LOG_LEVEL']}")
    print()

    try:
        # Run the master classifier as a subprocess
        print("🚀 Starting breast cancer classification project...")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, "master_breast_cancer_classifier.py"],
            env=env,
            capture_output=False,
            text=True,
        )

        print("=" * 60)

        if result.returncode == 0:
            print("✅ Project completed successfully!")
        else:
            print(f"❌ Project failed with return code: {result.returncode}")

    except Exception as e:
        print(f"❌ Error running project: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🏥 Breast Cancer Classification - Isolated Runner")
    print("=" * 60)

    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("✅ Multiprocessing start method set to 'spawn'")
    except RuntimeError as e:
        print(f"⚠️ Could not set start method: {e}")

    print()

    # Run the project
    success = run_isolated()

    if success:
        print("\n🎉 All tasks completed successfully!")
    else:
        print("\n❌ Project failed. Check error messages above.")

    input("\nPress Enter to exit...")
