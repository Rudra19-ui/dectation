# Breast Cancer Detection - Prediction System Guide

## 📋 Overview

This guide explains how to use the prediction system for breast cancer detection. The system can predict whether a mammogram image shows **Normal**, **Benign**, or **Malignant** tissue with confidence scores.

## 🚀 Quick Start

### 1. Basic Prediction

```python
from prediction_api import quick_predict, format_prediction_result

# Predict for a single image
result = quick_predict('path/to/mammogram.jpg')
print(format_prediction_result(result))
```

### 2. Using the Predictor Class

```python
from prediction_api import BreastCancerPredictor

# Initialize predictor
predictor = BreastCancerPredictor()

# Predict single image
result = predictor.predict('path/to/mammogram.jpg')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence_score']:.2%}")
```

### 3. Batch Prediction

```python
# Predict multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)

for result in results:
    print(format_prediction_result(result))
```

## 📊 Output Format

The prediction system returns a dictionary with the following information:

```python
{
    'predicted_class': 'Normal',  # or 'Benign', 'Malignant'
    'confidence_score': 0.85,     # Confidence between 0 and 1
    'all_probabilities': [0.85, 0.10, 0.05],  # Probabilities for all classes
    'success': True,              # Whether prediction was successful
    'original_image': <PIL.Image> # Original image object
}
```

## 🎯 Confidence Levels

- **🟢 High Confidence (>80%)**: Very reliable prediction
- **🟡 Medium Confidence (60-80%)**: Moderately reliable prediction  
- **🔴 Low Confidence (<60%)**: Less reliable prediction

## 📁 File Structure

```
project/
├── predict_model.py          # Core prediction functions
├── prediction_api.py         # Simple API for easy integration
├── test_prediction.py        # Test script with examples
├── models/
│   └── breast_cancer_detector_resnet50.pt  # Trained model
└── split_dataset/
    └── test/                 # Test images for validation
```

## 🔧 Model Requirements

### Supported Image Formats
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **Other formats** supported by PIL

### Image Processing
- Images are automatically resized to **224x224 pixels**
- Converted to RGB format
- Normalized using ImageNet statistics
- Preprocessed for ResNet50 architecture

## 📋 Complete Example

```python
#!/usr/bin/env python3
"""
Complete example of breast cancer prediction
"""

from prediction_api import BreastCancerPredictor, format_prediction_result
import os

def main():
    # Initialize predictor
    try:
        predictor = BreastCancerPredictor()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Example image path (replace with your image)
    image_path = "path/to/your/mammogram.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Make prediction
    print(f"🔍 Analyzing: {os.path.basename(image_path)}")
    result = predictor.predict(image_path)
    
    # Display results
    print("\n" + "="*50)
    print("📊 PREDICTION RESULTS")
    print("="*50)
    print(format_prediction_result(result))
    
    if result['success']:
        print(f"\n📋 Detailed Results:")
        print(f"   Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence_score']:.2%}")
        print(f"   Probabilities:")
        classes = ['Normal', 'Benign', 'Malignant']
        for class_name, prob in zip(classes, result['all_probabilities']):
            print(f"     {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()
```

## 🧪 Testing the System

Run the test script to verify everything works:

```bash
python test_prediction.py
```

This will:
- Load the trained model
- Test prediction on sample images
- Display results with visualizations
- Show confidence scores and probabilities

## 🔍 Advanced Usage

### Custom Model Path

```python
# Use a different model file
predictor = BreastCancerPredictor('path/to/custom_model.pt')
```

### Batch Processing

```python
# Process multiple images efficiently
image_paths = [
    'mammogram1.jpg',
    'mammogram2.jpg', 
    'mammogram3.jpg'
]

results = predictor.predict_batch(image_paths)

for i, result in enumerate(results):
    print(f"Image {i+1}: {format_prediction_result(result)}")
```

### Error Handling

```python
try:
    result = predictor.predict('image.jpg')
    if result['success']:
        print(f"Prediction: {result['predicted_class']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"System error: {e}")
```

## 📈 Model Performance

The trained model provides:
- **3-class classification**: Normal, Benign, Malignant
- **Confidence scores**: 0-100% reliability
- **Class probabilities**: Distribution across all classes
- **Visual results**: Images with prediction overlays

## ⚠️ Important Notes

1. **Model Requirements**: Ensure the trained model file exists in `models/`
2. **Image Quality**: Higher quality images generally provide better predictions
3. **Medical Disclaimer**: This is a research tool and should not replace professional medical diagnosis
4. **Confidence Threshold**: Consider predictions with <60% confidence as uncertain

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `models/breast_cancer_detector_resnet50.pt` exists
   - Train the model first if needed

2. **Image loading errors**
   - Check image format (PNG, JPEG supported)
   - Verify image file exists and is not corrupted

3. **Memory issues**
   - Use CPU if GPU memory is insufficient
   - Process images one at a time for large batches

### Getting Help

If you encounter issues:
1. Check the error messages for specific details
2. Verify all dependencies are installed
3. Ensure the model file exists and is not corrupted
4. Test with sample images from the test dataset

## 📞 Support

For technical support or questions about the prediction system, refer to the main project documentation or create an issue in the project repository. 