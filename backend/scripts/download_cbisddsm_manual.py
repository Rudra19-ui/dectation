#!/usr/bin/env python3
"""
Manual CBIS-DDSM Dataset Download Script
This script provides multiple methods to download the CBIS-DDSM dataset
"""

import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import requests


def create_download_structure():
    """Create the download directory structure"""
    print("Creating download directory structure...")

    # Create directories
    os.makedirs("dataset4/cbisddsm_download", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/original", exist_ok=True)

    print("✅ Directory structure created:")
    print("   dataset4/cbisddsm_download/  (for downloaded files)")
    print("   data/images/                 (for processed images)")
    print("   data/original/               (for original files)")


def download_sample_data():
    """Download a small sample dataset for testing"""
    print("\n📥 Downloading sample CBIS-DDSM data...")

    # Create sample data structure
    sample_dir = "dataset4/cbisddsm_sample"
    os.makedirs(sample_dir, exist_ok=True)

    # Create sample DICOM-like files (simplified for testing)
    sample_files = [
        {"name": "normal_001.dcm", "label": "normal"},
        {"name": "normal_002.dcm", "label": "normal"},
        {"name": "benign_001.dcm", "label": "benign"},
        {"name": "benign_002.dcm", "label": "benign"},
        {"name": "malignant_001.dcm", "label": "malignant"},
        {"name": "malignant_002.dcm", "label": "malignant"},
    ]

    print(f"Creating {len(sample_files)} sample files...")

    # Create sample files (this is a placeholder - real DICOM files would be downloaded)
    for i, file_info in enumerate(sample_files):
        # Create a simple text file as placeholder
        sample_file = os.path.join(sample_dir, file_info["name"])
        with open(sample_file, "w") as f:
            f.write(f"Sample DICOM file: {file_info['name']}\n")
            f.write(f"Label: {file_info['label']}\n")
            f.write("This is a placeholder for actual DICOM data\n")

    print("✅ Sample data created for testing")
    return sample_dir


def create_download_instructions():
    """Create comprehensive download instructions"""
    instructions = """# CBIS-DDSM Dataset Download Instructions

## Overview
The CBIS-DDSM dataset is a large mammography dataset (>10GB) that requires special tools to download.

## Method 1: TCIA Download Manager (Recommended)

1. **Download TCIA Download Manager**:
   - Go to: https://wiki.cancerimagingarchive.net/display/NBIA/Download+Manager+User+Guide
   - Download the appropriate version for your OS

2. **Import the manifest file**:
   - Open TCIA Download Manager
   - Import: `dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia`

3. **Set download directory**:
   - Set to: `dataset4/cbisddsm_download`

4. **Start download**:
   - This may take several hours depending on your connection

## Method 2: Manual Download from TCIA Website

1. **Visit the CBIS-DDSM page**:
   - Go to: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

2. **Download the dataset**:
   - Look for "Download" or "Data Access" section
   - Download the full dataset

3. **Extract to directory**:
   - Extract to: `dataset4/cbisddsm_download`

## Method 3: Using Python Scripts

If you have access to the NBIA Python client:

```bash
pip install nbia-data-retriever
nbia-data-retriever --manifest dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia --output dataset4/cbisddsm_download --include-annotations
```

## Method 4: Alternative Datasets

If CBIS-DDSM is too large, consider these alternatives:

1. **Mini-MIAS Dataset** (smaller, ~25MB):
   - Download from: http://peipa.essex.ac.uk/info/mias/
   - Contains 322 mammogram images

2. **INbreast Dataset** (medium size):
   - Requires registration at: https://www.breast-cancer-dataset.com/

## After Download

Once you have the dataset:

1. **Process the data**:
   ```bash
   python process_cbisddsm.py
   ```

2. **Train the model**:
   ```bash
   python train_model.py
   ```

## Dataset Information

- **CBIS-DDSM**: Curated Breast Imaging Subset of DDSM
- **Size**: >10GB
- **Format**: DICOM files
- **Labels**: Normal, Benign, Malignant
- **Views**: CC (craniocaudal) and MLO (mediolateral oblique)
- **Quality**: High-quality, professionally annotated

## Troubleshooting

- **Large download size**: Consider downloading during off-peak hours
- **Network issues**: Use a download manager with resume capability
- **Storage space**: Ensure you have at least 15GB free space
- **Processing time**: Converting DICOM to PNG may take 1-2 hours

## Next Steps

1. Download the dataset using one of the methods above
2. Run: `python process_cbisddsm.py`
3. Run: `python train_model.py`
4. Test the web app: `python -m streamlit run webapp/streamlit_app.py`
"""

    with open("CBIS-DDSM_DOWNLOAD_INSTRUCTIONS.md", "w", encoding="utf-8") as f:
        f.write(instructions)

    print("✅ Created comprehensive download instructions")


def create_sample_processing():
    """Create a sample processing script for testing"""
    print("\n🔧 Creating sample processing script...")

    sample_script = '''#!/usr/bin/env python3
"""
Sample CBIS-DDSM Processing Script
This script processes sample data for testing
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import glob

def create_sample_images(sample_dir, output_dir):
    """Create sample images from placeholder files"""
    print("Creating sample images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    sample_data = []
    
    for file_path in glob.glob(os.path.join(sample_dir, "*.dcm")):
        filename = os.path.basename(file_path)
        label = filename.split('_')[0]  # Extract label from filename
        
        # Create a sample image (white square with label text)
        img = Image.new('RGB', (224, 224), color='white')
        
        # Save the image
        output_filename = filename.replace('.dcm', '.png')
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)
        
        sample_data.append({
            'filename': output_filename,
            'label': label
        })
    
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
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save CSV files
    train_df[['filename', 'label']].to_csv('data/train.csv', index=False)
    val_df[['filename', 'label']].to_csv('data/val.csv', index=False)
    test_df[['filename', 'label']].to_csv('data/test.csv', index=False)
    
    print("Dataset splits created:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Show label distribution
    print("\\nLabel distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"   {split_name}:")
        label_counts = split_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"     {label}: {count}")
    
    print("\\nProcessing complete!")
    print("You can now run training with:")
    print("   python train_model.py")

if __name__ == "__main__":
    main()
'''

    with open("process_sample_cbisddsm.py", "w", encoding="utf-8") as f:
        f.write(sample_script)

    print("✅ Created sample processing script")


def main():
    """Main function"""
    print("CBIS-DDSM Dataset Download Setup")
    print("=" * 50)

    # Check manifest file
    manifest_file = "dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"

    if not os.path.exists(manifest_file):
        print(f"❌ Manifest file not found: {manifest_file}")
        return

    print(f"✅ Found manifest file: {manifest_file}")

    # Create directory structure
    create_download_structure()

    # Create download instructions
    create_download_instructions()

    # Create sample data for testing
    sample_dir = download_sample_data()

    # Create sample processing script
    create_sample_processing()

    print("\n🎉 Setup Complete!")
    print("\n📋 Available options:")
    print("1. Download full CBIS-DDSM dataset (see CBIS-DDSM_DOWNLOAD_INSTRUCTIONS.md)")
    print("2. Test with sample data:")
    print("   python process_sample_cbisddsm.py")
    print("   python train_model.py")

    print("\n📚 Dataset Information:")
    print("   - CBIS-DDSM is a large dataset (>10GB)")
    print("   - Contains mammogram images with annotations")
    print("   - Includes pathology information (normal, benign, malignant)")
    print("   - Multiple views: CC and MLO")
    print("   - High-quality DICOM format with metadata")

    print("\n⚠️  Note: Full download may take several hours")
    print("   Consider using the sample data for initial testing")


if __name__ == "__main__":
    main()
