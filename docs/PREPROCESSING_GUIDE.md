# 🖼️ Mammogram Image Preprocessing Guide

## 🎯 Overview
This guide helps you preprocess mammogram images for deep learning training with the following features:
- **Resize** all images to 224x224 pixels
- **Convert DICOM** to PNG format
- **Normalize** pixel values to [0, 1] range
- **Apply data augmentation** (rotation, zoom, flip, noise, blur, contrast)

## 🚀 Quick Start

### 1. Basic Preprocessing (Recommended)
```bash
python preprocess_images.py
```
This will:
- Use `organized_dataset/` as input
- Save to `preprocessed_dataset/`
- Resize to 224x224
- Convert DICOM files
- Apply data augmentation (2x factor)

### 2. Custom Configuration
```bash
python preprocess_images.py --input-dir organized_dataset --output-dir preprocessed_dataset --target-size 224 224 --augmentation-factor 3
```

### 3. Without Augmentation (Faster)
```bash
python preprocess_images.py --no-augmentation
```

### 4. Skip DICOM Conversion
```bash
python preprocess_images.py --no-dicom-convert
```

## 📊 What the Script Does

### 🔄 Image Processing Pipeline
1. **Load Images**: Supports PNG, JPEG, BMP, TIFF, DICOM formats
2. **Resize**: All images resized to 224x224 pixels
3. **Convert DICOM**: Medical DICOM files converted to PNG
4. **Normalize**: Pixel values normalized to [0, 1] range
5. **Data Augmentation**: Creates multiple versions with transformations

### 🎨 Data Augmentation Techniques
- **Rotation**: Random 90-degree rotations
- **Flip**: Horizontal and vertical flipping
- **Shift/Rotate**: Small translations and rotations
- **Noise**: Gaussian noise addition
- **Blur**: Gaussian blur effects
- **Contrast**: Brightness and contrast adjustments
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization

### 📁 Output Structure
```
preprocessed_dataset/
├── normal/          (original + augmented images)
├── benign/          (original + augmented images)
├── malignant/       (original + augmented images)
├── splits/
│   ├── train/       (preprocessed training data)
│   ├── val/         (preprocessed validation data)
│   └── test/        (preprocessed test data)
├── preprocessing_report.txt
└── preprocessing_visualization.png
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dir` | Input directory | `organized_dataset` |
| `--output-dir` | Output directory | `preprocessed_dataset` |
| `--target-size` | Image size (width height) | `224 224` |
| `--no-dicom-convert` | Skip DICOM conversion | `False` |
| `--no-augmentation` | Skip data augmentation | `False` |
| `--augmentation-factor` | Number of augmented versions | `2` |

## 📈 Expected Results

### Before Preprocessing:
- **Total Images**: 1,842
- **Formats**: Mixed (PNG, JPEG, DICOM)
- **Sizes**: Variable
- **Normalization**: None

### After Preprocessing:
- **Total Images**: ~5,526 (with 2x augmentation)
- **Format**: PNG (224x224)
- **Normalization**: [0, 1] range
- **Augmentation**: Multiple versions per image

## 🔧 Requirements

Install required packages:
```bash
pip install torch torchvision albumentations pydicom pillow numpy opencv-python matplotlib seaborn tqdm pandas
```

## 📊 Monitoring Progress

The script provides:
- **Progress bars** for each class
- **Real-time logging** of operations
- **Error tracking** and reporting
- **Final statistics** in report file
- **Visualization** of sample images

## 🎯 Use Cases

### 1. Training Data Preparation
```bash
python preprocess_images.py --augmentation-factor 3
```

### 2. Quick Testing
```bash
python preprocess_images.py --no-augmentation --target-size 128 128
```

### 3. Production Pipeline
```bash
python preprocess_images.py --input-dir /path/to/raw/data --output-dir /path/to/processed/data
```

## 📋 Output Files

### 1. Preprocessing Report (`preprocessing_report.txt`)
- Processing statistics
- Class distribution
- Error log
- Configuration summary

### 2. Visualization (`preprocessing_visualization.png`)
- Sample images from each class
- Before/after comparison
- Augmentation examples

### 3. Processed Images
- All images resized to 224x224
- Normalized pixel values
- Consistent PNG format
- Augmented versions (if enabled)

## 🚨 Troubleshooting

### Common Issues:
1. **Memory Error**: Reduce augmentation factor
2. **Slow Processing**: Use `--no-augmentation`
3. **DICOM Errors**: Use `--no-dicom-convert`
4. **File Not Found**: Check input directory path

### Performance Tips:
- Use SSD storage for faster I/O
- Increase RAM for large datasets
- Use GPU if available (automatic detection)
- Process in batches for very large datasets

## 🎉 Next Steps

After preprocessing:
1. **Train your model** with the preprocessed data
2. **Use the splits** for proper train/val/test evaluation
3. **Monitor performance** with the enhanced dataset
4. **Iterate** based on results

Your mammogram images are now ready for deep learning training! 🚀 