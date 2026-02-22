#!/usr/bin/env python3
"""
CBIS-DDSM Dataset Download and Processing Script
This script helps download and process the CBIS-DDSM mammography dataset
"""

import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import requests
from PIL import Image


def download_cbisddsm():
    """Download CBIS-DDSM dataset using the manifest file"""
    print("📥 Downloading CBIS-DDSM Dataset")
    print("=" * 50)

    # Check if manifest file exists
    manifest_file = "dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"

    if not os.path.exists(manifest_file):
        print("❌ Manifest file not found!")
        print("Please ensure the CBIS-DDSM manifest file is in dataset4/ directory")
        return False

    print("✅ Found CBIS-DDSM manifest file")
    print("📋 This dataset contains:")
    print("   - Mammogram images (DICOM format)")
    print("   - Annotations and labels")
    print("   - Multiple views (CC, MLO)")
    print("   - Pathology information")

    # Create download directory
    download_dir = "dataset4/cbisddsm_download"
    os.makedirs(download_dir, exist_ok=True)

    print(f"\n📁 Download directory: {download_dir}")
    print("⚠️  Note: This is a large dataset (>10GB)")
    print("   Download may take several hours depending on your connection")

    return True


def process_cbisddsm_data():
    """Process downloaded CBIS-DDSM data into our format"""
    print("\n🔄 Processing CBIS-DDSM Data")
    print("=" * 50)

    # Expected structure after download
    data_dir = "dataset4/cbisddsm_download"

    if not os.path.exists(data_dir):
        print("❌ Download directory not found!")
        print("Please download the CBIS-DDSM dataset first")
        return False

    # Create our data structure
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/original", exist_ok=True)

    print("📁 Creating data structure...")
    print("   data/")
    print("   ├── images/     (processed images)")
    print("   ├── original/   (original DICOM files)")
    print("   ├── train.csv   (training data)")
    print("   ├── val.csv     (validation data)")
    print("   └── test.csv    (test data)")

    return True


def create_sample_processing():
    """Create a sample processing script for CBIS-DDSM"""
    print("\n🔧 Creating CBIS-DDSM Processing Script")

    script_content = '''#!/usr/bin/env python3
"""
CBIS-DDSM Data Processing Script
This script processes downloaded CBIS-DDSM DICOM files into our training format
"""

import os
import pandas as pd
import pydicom
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import glob

def process_dicom_to_image(dicom_path, output_path, size=(224, 224)):
    """Convert DICOM file to PNG image"""
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(dicom_path)
        
        # Get pixel data
        pixel_array = dcm.pixel_array
        
        # Normalize to 0-255
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(pixel_array)
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save as PNG
        img.save(output_path, 'PNG')
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
            pathology = getattr(dcm, 'Pathology', 'Unknown')
            
            # Map to our classes
            if 'BENIGN' in str(pathology).upper():
                label = 'benign'
            elif 'MALIGNANT' in str(pathology).upper():
                label = 'malignant'
            else:
                label = 'normal'
            
            labels.append({
                'filename': os.path.basename(dicom_file).replace('.dcm', '.png'),
                'label': label,
                'dicom_path': dicom_file
            })
            
        except Exception as e:
            print(f"Error reading metadata from {dicom_file}: {e}")
    
    return labels

def main():
    """Main processing function"""
    print("🏥 CBIS-DDSM Data Processing")
    print("=" * 50)
    
    # Configuration
    dicom_dir = "dataset4/cbisddsm_download"  # Your downloaded DICOM files
    output_dir = "data/images"
    
    if not os.path.exists(dicom_dir):
        print(f"❌ DICOM directory not found: {dicom_dir}")
        print("Please download CBIS-DDSM dataset first")
        return
    
    print(f"📁 Processing DICOM files from: {dicom_dir}")
    
    # Extract labels from metadata
    print("🔍 Extracting labels from DICOM metadata...")
    labels = extract_labels_from_metadata(dicom_dir)
    
    if not labels:
        print("❌ No valid DICOM files found!")
        return
    
    print(f"✅ Found {len(labels)} DICOM files")
    
    # Process images
    print("🖼️ Converting DICOM to PNG images...")
    processed_count = 0
    
    for item in labels:
        dicom_path = item['dicom_path']
        output_path = os.path.join(output_dir, item['filename'])
        
        if process_dicom_to_image(dicom_path, output_path):
            processed_count += 1
    
    print(f"✅ Successfully processed {processed_count}/{len(labels)} images")
    
    # Create CSV files
    if processed_count > 0:
        print("📊 Creating dataset splits...")
        
        # Create DataFrame
        df = pd.DataFrame([item for item in labels if os.path.exists(
            os.path.join(output_dir, item['filename']))])
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        
        # Save CSV files
        train_df[['filename', 'label']].to_csv('data/train.csv', index=False)
        val_df[['filename', 'label']].to_csv('data/val.csv', index=False)
        test_df[['filename', 'label']].to_csv('data/test.csv', index=False)
        
        print("✅ Dataset splits created:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Validation: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        
        # Show label distribution
        print("\n📊 Label distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"   {split_name}:")
            label_counts = split_df['label'].value_counts()
            for label, count in label_counts.items():
                print(f"     {label}: {count}")
    
    print("\n🎉 Processing complete!")
    print("📋 You can now run training with:")
    print("   python train_model.py")

if __name__ == "__main__":
    main()
'''

    with open("process_cbisddsm.py", "w") as f:
        f.write(script_content)

    print("✅ Created process_cbisddsm.py")
    print("📋 Next steps:")
    print("1. Download CBIS-DDSM dataset using the manifest file")
    print("2. Run: python process_cbisddsm.py")
    print("3. Run: python train_model.py")


def main():
    """Main function"""
    print("🏥 CBIS-DDSM Dataset Setup")
    print("=" * 50)

    # Check manifest file
    if download_cbisddsm():
        # Process data structure
        if process_cbisddsm_data():
            # Create processing script
            create_sample_processing()

            print("\n🎉 Setup Complete!")
            print("📋 To use CBIS-DDSM dataset:")
            print("1. Download the dataset using the manifest file")
            print("2. Run: python process_cbisddsm.py")
            print("3. Run: python train_model.py")

            print("\n📚 Dataset Information:")
            print("   - CBIS-DDSM is a curated subset of DDSM")
            print("   - Contains mammogram images with annotations")
            print("   - Includes pathology information (normal, benign, malignant)")
            print(
                "   - Multiple views: CC (craniocaudal) and MLO (mediolateral oblique)"
            )
            print("   - High-quality DICOM format with metadata")
        else:
            print("❌ Failed to process data structure")
    else:
        print("❌ Failed to setup download")


if __name__ == "__main__":
    main()
