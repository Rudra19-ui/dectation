# DICOM to PNG Converter for Mammogram Images

A comprehensive Python script to convert DICOM (.dcm) files to PNG format, specifically optimized for mammogram images with proper windowing, normalization, and enhancement.

## 🎯 Features

- ✅ **Batch Processing**: Convert multiple DICOM files at once
- ✅ **Mammogram Optimization**: Specialized windowing and enhancement for mammogram images
- ✅ **Recursive Search**: Finds DICOM files in subfolders automatically
- ✅ **Error Handling**: Robust error handling with detailed reporting
- ✅ **DICOM Information**: Extract and display DICOM metadata
- ✅ **Image Enhancement**: CLAHE, noise reduction, and edge enhancement
- ✅ **Flexible Output**: Customizable output folder and naming

## 📋 Requirements

### Required Packages
```bash
pip install pydicom opencv-python numpy matplotlib
```

### Optional Packages
```bash
pip install pillow  # For additional image processing
```

## 🚀 Quick Start

### Basic Usage

```bash
# Convert all DICOM files in a folder
python convert_dicom_to_png.py "path/to/dicom/folder"

# Convert with custom output folder
python convert_dicom_to_png.py "path/to/dicom/folder" -o "path/to/output/folder"

# Convert with custom window settings
python convert_dicom_to_png.py "path/to/dicom/folder" -wc 1500 -ww 3000
```

### Advanced Usage

```bash
# Test with limited files
python convert_dicom_to_png.py "path/to/dicom/folder" -m 5

# Show DICOM information for a specific file
python convert_dicom_to_png.py "path/to/dicom/folder" --info "sample.dcm"

# Convert without progress information
python convert_dicom_to_png.py "path/to/dicom/folder" --no-progress
```

## 📁 File Structure

### Input Structure
```
dicom_folder/
├── patient1/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── subfolder/
│       └── image3.dcm
├── patient2/
│   └── image4.dcm
└── other_files.dcm
```

### Output Structure
```
converted_png/
├── image1.png
├── image2.png
├── image3.png
├── image4.png
└── other_files.png
```

## 🔧 Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `input_folder` | Path to folder containing DICOM files | `"E:\mammograms"` |
| `-o, --output` | Custom output folder | `-o "E:\converted"` |
| `-wc, --window-center` | Window center for display | `-wc 1500` |
| `-ww, --window-width` | Window width for display | `-ww 3000` |
| `-m, --max-files` | Limit number of files (testing) | `-m 10` |
| `--info` | Show DICOM info for specific file | `--info "sample.dcm"` |
| `--no-progress` | Hide progress information | `--no-progress` |

## 🏥 Mammogram-Specific Features

### Window/Level Settings
The script uses optimized window/level settings for mammogram images:

- **Default Window Center**: 1500
- **Default Window Width**: 3000
- **Custom Settings**: Can be specified via command line

### Image Enhancement
1. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
2. **Noise Reduction**: Gaussian blur to reduce noise
3. **Edge Enhancement**: Unsharp masking for better detail

### DICOM Information Extraction
- Patient information (anonymized)
- Image modality and size
- Manufacturer information
- Window/level settings

## 📊 Example Output

### Console Output
```
============================================================
🏥 DICOM to PNG Converter for Mammogram Images
============================================================
🔍 Found 25 DICOM files
📁 Input folder: E:\mammograms
📁 Output folder: E:\mammograms\converted_png
============================================================

📄 Processing 1/25: patient001_image1.dcm
✅ Converted: patient001_image1.dcm -> patient001_image1.png
   Size: (2048, 1024), Modality: MG

📄 Processing 2/25: patient001_image2.dcm
✅ Converted: patient001_image2.dcm -> patient001_image2.png
   Size: (2048, 1024), Modality: MG

...

============================================================
📊 CONVERSION SUMMARY
============================================================
📁 Input folder: E:\mammograms
📁 Output folder: E:\mammograms\converted_png
📄 Total files found: 25
✅ Successful conversions: 23
❌ Failed conversions: 1
⚠️ Skipped files: 1

📈 Success rate: 92.0%
🎉 Conversion completed successfully!
📁 Check output folder: E:\mammograms\converted_png
```

### DICOM Information Output
```
📋 DICOM Information for: sample.dcm
----------------------------------------
patient_name: Anonymous
patient_id: 12345
study_date: 20231201
modality: MG
image_size: (2048, 1024)
bits_allocated: 16
samples_per_pixel: 1
window_center: 1500
window_width: 3000
manufacturer: GE MEDICAL SYSTEMS
manufacturer_model: Senographe DS

📊 Pixel Array Information:
Shape: (2048, 1024)
Data type: uint16
Min value: 0
Max value: 4095
Mean value: 2047.50
```

## 🔍 DICOM Information Tool

### View DICOM Metadata
```bash
python convert_dicom_to_png.py "path/to/folder" --info "sample.dcm"
```

This will display:
- Patient information (anonymized)
- Image properties (size, modality, etc.)
- Window/level settings
- Manufacturer information
- Pixel array statistics

## 🛠️ Technical Details

### Window/Level Transformation
```python
# Apply window/level transformation
window_min = window_center - window_width // 2
window_max = window_center + window_width // 2
windowed = np.clip(pixel_array, window_min, window_max)
normalized = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
```

### Image Enhancement Pipeline
1. **CLAHE**: Improves local contrast
2. **Gaussian Blur**: Reduces noise (3x3 kernel)
3. **Unsharp Masking**: Enhances edges

### Supported DICOM Formats
- Standard DICOM files (.dcm extension)
- DICOM files without .dcm extension
- Various bit depths (8-bit, 12-bit, 16-bit)
- Different modalities (MG, CR, DX, etc.)

## 🚨 Troubleshooting

### Common Issues

#### 1. **"No DICOM files found"**
```bash
# Check if files exist
dir "path/to/folder" /s *.dcm

# Try with different file extensions
python convert_dicom_to_png.py "path/to/folder" --info "filename"
```

#### 2. **"Invalid DICOM file"**
- File might be corrupted
- File might not be DICOM format
- Try with `--info` to check file details

#### 3. **"Memory error"**
```bash
# Limit number of files for testing
python convert_dicom_to_png.py "path/to/folder" -m 5
```

#### 4. **"Permission denied"**
- Check folder permissions
- Run as administrator if needed
- Ensure output folder is writable

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `InvalidDicomError` | File is not DICOM format | Check file format |
| `PermissionError` | No write permission | Check folder permissions |
| `MemoryError` | Insufficient memory | Reduce batch size or close other apps |
| `FileNotFoundError` | Input folder doesn't exist | Check folder path |

## 📈 Performance Tips

### For Large Datasets
1. **Test First**: Use `-m 5` to test with few files
2. **Close Applications**: Free up memory
3. **Use SSD**: Faster I/O for large files
4. **Monitor Progress**: Use `--no-progress` for faster processing

### For Different Image Types
- **Mammograms**: Use default settings (wc=1500, ww=3000)
- **Chest X-rays**: Try wc=1000, ww=2000
- **CT Images**: Try wc=50, ww=400

## 🔒 Privacy and Security

### Patient Information
- Patient names are anonymized in output
- Original DICOM files are not modified
- No patient data is logged or stored

### File Handling
- Original DICOM files remain unchanged
- PNG files contain only image data
- No metadata is embedded in PNG files

## 📝 Usage Examples

### Example 1: Basic Conversion
```bash
# Convert all DICOM files in current folder
python convert_dicom_to_png.py "."

# Output: ./converted_png/
```

### Example 2: Custom Output Folder
```bash
# Convert to specific output folder
python convert_dicom_to_png.py "E:\mammograms" -o "E:\converted_images"

# Output: E:\converted_images\
```

### Example 3: Custom Window Settings
```bash
# Use custom window/level for better contrast
python convert_dicom_to_png.py "E:\mammograms" -wc 1200 -ww 2500
```

### Example 4: Testing Mode
```bash
# Test with first 3 files
python convert_dicom_to_png.py "E:\mammograms" -m 3
```

### Example 5: DICOM Information
```bash
# View DICOM metadata
python convert_dicom_to_png.py "." --info "sample.dcm"
```

## 🎯 Integration with Training Pipeline

### For Machine Learning
```bash
# Convert DICOM files for training
python convert_dicom_to_png.py "E:\raw_dicom" -o "E:\training_data"

# Use converted PNG files in training script
python windows_training_guide.py
```

### Batch Processing
```bash
# Process multiple folders
for folder in "folder1" "folder2" "folder3"; do
    python convert_dicom_to_png.py "$folder" -o "converted_$folder"
done
```

## 📞 Support

### Getting Help
1. **Check DICOM Information**: Use `--info` to verify file format
2. **Test with Few Files**: Use `-m 5` to test configuration
3. **Check Error Messages**: Review console output for specific errors
4. **Verify File Paths**: Ensure input/output folders exist

### Common Commands
```bash
# Check if script works
python convert_dicom_to_png.py --help

# Test with one file
python convert_dicom_to_png.py "folder" -m 1

# View DICOM info
python convert_dicom_to_png.py "folder" --info "file.dcm"
```

## 🎉 Success!

Once conversion is complete:
1. **Check Output Folder**: Verify PNG files were created
2. **Review Summary**: Check success rate and error messages
3. **Test Images**: Open a few PNG files to verify quality
4. **Proceed with Training**: Use converted images in your ML pipeline

The converted PNG files are now ready for use in your breast cancer classification training pipeline! 