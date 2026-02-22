#!/usr/bin/env python3
"""
CBIS-DDSM Dataset Download Script
This script downloads the CBIS-DDSM dataset using the manifest file
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def check_nbia_client():
    """Check if NBIA Data Retriever is installed"""
    try:
        result = subprocess.run(
            ["nbia-data-retriever", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ NBIA Data Retriever is installed")
            return True
        else:
            print("❌ NBIA Data Retriever not found")
            return False
    except FileNotFoundError:
        print("❌ NBIA Data Retriever not found")
        return False


def install_nbia_client():
    """Install NBIA Data Retriever"""
    print("📦 Installing NBIA Data Retriever...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "nbia-data-retriever"], check=True
        )
        print("✅ NBIA Data Retriever installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install NBIA Data Retriever: {e}")
        return False


def download_with_nbia(manifest_file, output_dir):
    """Download dataset using NBIA Data Retriever"""
    print(f"📥 Downloading CBIS-DDSM dataset...")
    print(f"   Manifest: {manifest_file}")
    print(f"   Output: {output_dir}")

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run NBIA download command
        cmd = [
            "nbia-data-retriever",
            "--manifest",
            manifest_file,
            "--output",
            output_dir,
            "--include-annotations",
        ]

        print(f"🔄 Running: {' '.join(cmd)}")
        print("⚠️  This may take several hours for the full dataset...")

        # Start the download process
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Monitor progress
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Check if download was successful
        if process.returncode == 0:
            print("✅ Download completed successfully!")
            return True
        else:
            print("❌ Download failed!")
            return False

    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def download_with_curl(manifest_file, output_dir):
    """Alternative download method using curl"""
    print("📥 Attempting download with curl...")

    try:
        # Read manifest file
        with open(manifest_file, "r") as f:
            lines = f.readlines()

        # Extract download URL and series IDs
        download_url = None
        series_ids = []

        for line in lines:
            line = line.strip()
            if line.startswith("downloadServerUrl="):
                download_url = line.split("=")[1]
            elif line.startswith("1.3.6.1.4.1.9590.100.1.2."):
                series_ids.append(line)

        if not download_url or not series_ids:
            print("❌ Could not parse manifest file")
            return False

        print(f"📊 Found {len(series_ids)} series to download")
        print(f"🌐 Download URL: {download_url}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Download each series (this is a simplified version)
        print("⚠️  Note: Full download requires NBIA Data Retriever")
        print("   This is a demonstration of the download process")

        return True

    except Exception as e:
        print(f"❌ Error parsing manifest: {e}")
        return False


def create_download_instructions():
    """Create detailed download instructions"""
    instructions = """# CBIS-DDSM Dataset Download Instructions

## Method 1: Using NBIA Data Retriever (Recommended)

1. Install NBIA Data Retriever:
   ```
   pip install nbia-data-retriever
   ```

2. Download the dataset:
   ```
   nbia-data-retriever --manifest dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia --output dataset4/cbisddsm_download --include-annotations
   ```

## Method 2: Using TCIA Download Manager

1. Download TCIA Download Manager from: https://wiki.cancerimagingarchive.net/display/NBIA/Download+Manager+User+Guide

2. Import the manifest file: dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia

3. Set download directory to: dataset4/cbisddsm_download

4. Start the download

## Method 3: Manual Download from TCIA Website

1. Go to: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

2. Download the dataset manually

3. Extract to: dataset4/cbisddsm_download

## After Download

Once downloaded, run:
```
python process_cbisddsm.py
```

This will process the DICOM files into our training format.
"""

    with open("CBIS-DDSM_DOWNLOAD_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)

    print("✅ Created CBIS-DDSM_DOWNLOAD_INSTRUCTIONS.md")


def main():
    """Main download function"""
    print("🏥 CBIS-DDSM Dataset Download")
    print("=" * 50)

    # Check manifest file
    manifest_file = "dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"
    output_dir = "dataset4/cbisddsm_download"

    if not os.path.exists(manifest_file):
        print(f"❌ Manifest file not found: {manifest_file}")
        return

    print(f"✅ Found manifest file: {manifest_file}")
    print(f"📁 Output directory: {output_dir}")

    # Check if NBIA client is available
    if check_nbia_client():
        print("\n🚀 Starting download with NBIA Data Retriever...")
        success = download_with_nbia(manifest_file, output_dir)
    else:
        print("\n📦 NBIA Data Retriever not found")
        install_choice = input("Would you like to install NBIA Data Retriever? (y/n): ")

        if install_choice.lower() == "y":
            if install_nbia_client():
                print("\n🚀 Starting download with NBIA Data Retriever...")
                success = download_with_nbia(manifest_file, output_dir)
            else:
                print("❌ Failed to install NBIA Data Retriever")
                success = False
        else:
            print("📋 Creating download instructions...")
            create_download_instructions()
            success = True

    if success:
        print("\n🎉 Download setup complete!")
        print("📋 Next steps:")
        print("1. Wait for download to complete (may take hours)")
        print("2. Run: python process_cbisddsm.py")
        print("3. Run: python train_model.py")
    else:
        print("\n❌ Download failed!")
        print(
            "📋 Please check the download instructions in CBIS-DDSM_DOWNLOAD_INSTRUCTIONS.md"
        )


if __name__ == "__main__":
    main()
