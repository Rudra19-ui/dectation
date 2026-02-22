# 📊 Dataset Splitting Guide

## 🎯 Overview
This guide helps you split your mammogram dataset into balanced train, validation, and test sets:
- **70% Training** - For model training
- **15% Validation** - For hyperparameter tuning
- **15% Testing** - For final evaluation

Ensures balanced class distribution across all splits using stratified sampling.

## 🚀 Quick Start

### 1. Basic Splitting (Recommended)
```bash
python split_dataset.py
```
This will:
- Use `preprocessed_dataset/` as input
- Save to `split_dataset/`
- Split with 70/15/15 ratios
- Maintain class balance

### 2. Custom Split Ratios
```bash
python split_dataset.py --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10
```

### 3. Custom Directories
```bash
python split_dataset.py --input-dir my_dataset --output-dir my_splits
```

### 4. Reproducible Splits
```bash
python split_dataset.py --random-state 123
```

## 📊 What the Script Does

### 🔄 Splitting Process
1. **Load Images**: Reads all images from each class directory
2. **Stratified Sampling**: Maintains class proportions in each split
3. **Random Shuffling**: Ensures unbiased distribution
4. **File Copying**: Creates organized split directories
5. **Validation**: Checks for balanced distribution

### 📁 Input Structure
```
preprocessed_dataset/
├── normal/          (810 images)
├── benign/          (2,736 images)
├── malignant/       (2,478 images)
└── splits/          (existing splits, optional)
```

### 📁 Output Structure
```
split_dataset/
├── train/
│   ├── normal/      (70% of normal images)
│   ├── benign/      (70% of benign images)
│   └── malignant/   (70% of malignant images)
├── val/
│   ├── normal/      (15% of normal images)
│   ├── benign/      (15% of benign images)
│   └── malignant/   (15% of malignant images)
├── test/
│   ├── normal/      (15% of normal images)
│   ├── benign/      (15% of benign images)
│   └── malignant/   (15% of malignant images)
├── dataset_splits.csv
├── split_summary.csv
├── split_report.txt
├── split_distribution.png
└── class_distribution.png
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Input directory | `preprocessed_dataset` |
| `--output-dir` | Output directory | `split_dataset` |
| `--train-ratio` | Training set proportion | `0.70` |
| `--val-ratio` | Validation set proportion | `0.15` |
| `--test-ratio` | Test set proportion | `0.15` |
| `--random-state` | Random seed | `42` |
| `--no-preserve-splits` | Skip existing splits | `False` |

## 📈 Expected Results

### Before Splitting:
- **Total Images**: 6,024 (from preprocessing)
- **Normal**: 810 images
- **Benign**: 2,736 images
- **Malignant**: 2,478 images

### After Splitting (70/15/15):
- **Training**: ~4,217 images (70%)
- **Validation**: ~904 images (15%)
- **Testing**: ~904 images (15%)

### Class Distribution (Maintained):
- **Normal**: ~567 train, 122 val, 122 test
- **Benign**: ~1,915 train, 410 val, 410 test
- **Malignant**: ~1,735 train, 372 val, 372 test

## 🔧 Requirements

Install required packages:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

## 📊 Monitoring Progress

The script provides:
- **Progress bars** for each class and split
- **Real-time logging** of operations
- **Split statistics** for each class
- **Validation** of balanced distribution
- **Visualizations** of split distribution

## 🎯 Use Cases

### 1. Standard Split
```bash
python split_dataset.py
```

### 2. Larger Training Set
```bash
python split_dataset.py --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10
```

### 3. Reproducible Research
```bash
python split_dataset.py --random-state 42
```

### 4. Custom Dataset
```bash
python split_dataset.py --input-dir /path/to/dataset --output-dir /path/to/splits
```

## 📋 Output Files

### 1. Split Directories
- **train/**: Training images organized by class
- **val/**: Validation images organized by class
- **test/**: Test images organized by class

### 2. CSV Files
- **dataset_splits.csv**: Complete file listing with paths
- **split_summary.csv**: Summary statistics by split and class

### 3. Report Files
- **split_report.txt**: Detailed split statistics and validation
- **split_distribution.png**: Visualization of split distribution
- **class_distribution.png**: Overall class distribution pie chart

## 🚨 Troubleshooting

### Common Issues:
1. **Memory Error**: Process smaller batches
2. **File Not Found**: Check input directory path
3. **Imbalanced Splits**: Check class distribution in input
4. **Permission Error**: Check write permissions for output directory

### Validation Checks:
- Ensures no class has 0 images in any split
- Warns about extreme class imbalance (>10x ratio)
- Validates split ratios sum to 1.0
- Checks file copying success

## 🎉 Next Steps

After splitting:
1. **Train your model** using the train split
2. **Tune hyperparameters** using the validation split
3. **Evaluate performance** using the test split
4. **Monitor overfitting** by comparing train/val performance

## 📊 Split Statistics Example

```
DATASET SPLIT REPORT
==================================================

Input Directory: preprocessed_dataset
Output Directory: split_dataset
Train Ratio: 70.0%
Validation Ratio: 15.0%
Test Ratio: 15.0%
Random State: 42

SPLIT STATISTICS:
------------------------------
Total Images: 6024
Errors: 0

CLASS DISTRIBUTION:
--------------------
Normal: 810 images
Benign: 2736 images
Malignant: 2478 images

SPLIT BREAKDOWN:
--------------------

TRAIN SET:
  Normal: 567 images
  Benign: 1915 images
  Malignant: 1735 images
  Total: 4217 images

VAL SET:
  Normal: 122 images
  Benign: 410 images
  Malignant: 372 images
  Total: 904 images

TEST SET:
  Normal: 122 images
  Benign: 410 images
  Malignant: 372 images
  Total: 904 images
```

Your dataset is now perfectly split and ready for training! 🚀 