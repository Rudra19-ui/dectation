# 📁 Dataset Organization Guide

## 🎯 Overview
This guide helps you organize mammogram datasets into three categories:
- **normal** - Healthy mammograms without abnormalities
- **benign** - Mammograms with benign (non-cancerous) findings
- **malignant** - Mammograms with malignant (cancerous) findings

## 🚀 Quick Start

### 1. Basic Organization (Recommended)
```bash
python organize_dataset.py
```
This will:
- Use your existing enhanced dataset (`data/train_enhanced.csv`)
- Organize images from `data/images/`
- Create folders: `organized_dataset/normal/`, `organized_dataset/benign/`, `organized_dataset/malignant/`

### 2. Full Organization with CBIS-DDSM
```bash
python organize_dataset.py --create-splits --convert-dicom
```
This will:
- Organize both enhanced and CBIS-DDSM datasets
- Convert DICOM files to PNG format
- Create train/validation/test splits

### 3. Custom Configuration
```bash
python organize_dataset.py \
    --output-dir "my_organized_data" \
    --enhanced-csv "data/train_enhanced.csv" \
    --enhanced-images "data/images" \
    --cbisddsm-csv "dataset4/mass_case_description_train_set.csv" \
    --cbisddsm-base "dataset4" \
    --create-splits \
    --convert-dicom
```

## 📊 Expected Output Structure

```
organized_dataset/
├── normal/
│   ├── normal_001.png
│   ├── normal_002.png
│   └── ...
├── benign/
│   ├── benign_001.png
│   ├── benign_002.png
│   └── ...
├── malignant/
│   ├── malignant_001.png
│   ├── malignant_002.png
│   └── ...
├── splits/
│   ├── train/
│   │   ├── normal/
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/
│   │   ├── normal/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── normal/
│       ├── benign/
│       └── malignant/
├── dataset_summary.txt
└── converted_png/ (if DICOM conversion enabled)
```

## 🔧 Supported File Formats

### Input Formats
- **PNG** - Portable Network Graphics
- **JPEG/JPG** - Joint Photographic Experts Group
- **DICOM** - Digital Imaging and Communications in Medicine

### Output Formats
- **PNG** - All images converted to PNG for consistency

## 📋 Labeling Methods

### 1. CSV-Based Labeling (Enhanced Dataset)
```csv
filename,label,original_file
normal_001.png,normal,normal (1).png
benign_001.png,benign,benign (1).png
malignant_001.png,malignant,malignant (1).png
```

### 2. Pathology-Based Labeling (CBIS-DDSM)
```csv
patient_id,pathology,image_path
P_00001,MALIGNANT,Mass-Training_P_00001_LEFT_CC/...
P_00004,BENIGN,Mass-Training_P_00004_LEFT_CC/...
```

### 3. Filename Pattern Labeling (Fallback)
- Files containing "normal" → normal category
- Files containing "benign" → benign category  
- Files containing "malignant" → malignant category

## 🛠️ Advanced Features

### DICOM to PNG Conversion
```bash
python organize_dataset.py --convert-dicom
```
- Converts DICOM files to PNG format
- Preserves image quality and metadata
- Handles various DICOM formats

### Train/Validation/Test Splits
```bash
python organize_dataset.py --create-splits
```
- Creates 70% train, 15% validation, 15% test splits
- Maintains class balance in each split
- Generates split summary reports

### Custom Split Ratios
```python
# In the script, modify these values:
train_ratio = 0.7    # 70% for training
val_ratio = 0.15     # 15% for validation  
test_ratio = 0.15    # 15% for testing
```

## 📈 Dataset Statistics

After organization, you'll get a summary like:
```
DATASET ORGANIZATION SUMMARY
========================================

Total Images: 1198

Normal: 133 images
Benign: 654 images
Malignant: 411 images

Errors: 0
```

## 🔍 Troubleshooting

### Common Issues

1. **"File not found" errors**
   - Check file paths in CSV files
   - Ensure images exist in specified directories
   - Verify file permissions

2. **"Unknown label" warnings**
   - Check label spelling in CSV files
   - Ensure labels match: normal, benign, malignant
   - Review filename patterns

3. **DICOM conversion errors**
   - Install pydicom: `pip install pydicom`
   - Check DICOM file integrity
   - Verify file permissions

### Error Log
The script creates an error log in `dataset_summary.txt` with details about any issues encountered during organization.

## 📝 Manual Organization (Alternative)

If you prefer manual organization:

### Step 1: Create Folders
```bash
mkdir -p organized_dataset/{normal,benign,malignant}
```

### Step 2: Copy Files by Category
```bash
# Using filename patterns
cp data/images/normal_*.png organized_dataset/normal/
cp data/images/benign_*.png organized_dataset/benign/
cp data/images/malignant_*.png organized_dataset/malignant/
```

### Step 3: Verify Organization
```bash
# Count files in each category
ls organized_dataset/normal/ | wc -l
ls organized_dataset/benign/ | wc -l
ls organized_dataset/malignant/ | wc -l
```

## 🎯 Best Practices

1. **Backup Original Data**
   - Always keep original files as backup
   - Use copy operations, not move operations

2. **Verify Labels**
   - Double-check CSV file accuracy
   - Review filename patterns
   - Validate pathology information

3. **Maintain Consistency**
   - Use consistent file naming
   - Standardize image formats
   - Keep organized folder structure

4. **Document Changes**
   - Note any manual corrections
   - Record data sources
   - Document labeling decisions

## 📞 Support

If you encounter issues:
1. Check the error log in `dataset_summary.txt`
2. Verify file paths and permissions
3. Ensure all required dependencies are installed
4. Review the troubleshooting section above

## 🔄 Next Steps

After organization:
1. **Train your model** with the organized dataset
2. **Validate data quality** by reviewing sample images
3. **Create data loaders** for your training pipeline
4. **Monitor class balance** during training

---

*This guide helps you create a well-organized dataset for breast cancer detection models.* 