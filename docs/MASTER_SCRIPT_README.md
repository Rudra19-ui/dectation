# Master Breast Cancer Classification Script

A comprehensive, all-in-one Python script that handles the complete breast cancer classification pipeline: **Load Dataset → Train Model → Predict → Evaluate**. Optimized for Windows compatibility with proper function structure.

## 🎯 Features

- ✅ **Complete Pipeline**: Single script handles everything from data loading to evaluation
- ✅ **Windows Optimized**: No multiprocessing issues, stable training
- ✅ **Smart Model Management**: Loads existing model or trains new one automatically
- ✅ **Transfer Learning**: Uses ResNet50V2 pre-trained on ImageNet
- ✅ **Comprehensive Evaluation**: Accuracy, confusion matrix, classification report
- ✅ **Sample Predictions**: Tests on random samples from each class
- ✅ **Visualization**: Training history and confusion matrix plots
- ✅ **Results Export**: Saves all results to CSV, PNG, and text files
- ✅ **Error Handling**: Robust error handling with detailed reporting

## 📋 Requirements

### Required Packages
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pandas
```

### Dataset Structure
```
E:\rudra\project\dataset\
├── benign\
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── malignant\
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── normal\
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## 🚀 Quick Start

### Basic Usage
```bash
# Run the complete pipeline
python master_breast_cancer_classifier.py
```

### Custom Configuration
```python
# Create classifier with custom settings
classifier = BreastCancerMasterClassifier(
    dataset_path="E:\\rudra\\project\\dataset",
    model_path="my_model.h5"
)

# Run pipeline with custom number of samples
classifier.run_complete_pipeline(num_samples=10)
```

## 📊 What the Script Does

### 1. **Dataset Loading & Verification**
- ✅ Checks dataset structure (`benign`, `malignant`, `normal` folders)
- ✅ Counts images in each class
- ✅ Validates file formats (JPG, PNG)

### 2. **Model Management**
- ✅ **Loads existing model** if available (`breast_cancer_master_model.h5`)
- ✅ **Trains new model** if no existing model found
- ✅ **Transfer learning** with ResNet50V2
- ✅ **Windows-safe training** (no multiprocessing issues)

### 3. **Sample Predictions**
- ✅ **Random sampling** from each class
- ✅ **Confidence scores** for each prediction
- ✅ **Detailed results** with file names and probabilities

### 4. **Comprehensive Evaluation**
- ✅ **Accuracy calculation**
- ✅ **Confusion matrix** visualization
- ✅ **Classification report** (precision, recall, F1-score)
- ✅ **Training history** plots (if training occurred)

### 5. **Results Export**
- ✅ **Predictions CSV** with detailed results
- ✅ **Confusion matrix CSV** and PNG
- ✅ **Classification report** text file
- ✅ **Summary file** with all statistics
- ✅ **Training history** plots (PNG)

## 🔧 Configuration Options

### Class Initialization
```python
classifier = BreastCancerMasterClassifier(
    dataset_path="E:\\rudra\\project\\dataset",  # Dataset location
    model_path="breast_cancer_master_model.h5"   # Model save/load path
)
```

### Training Configuration
```python
# Default settings (can be modified in the class)
self.epochs = 30                    # Training epochs
self.learning_rate = 1e-4          # Learning rate
self.batch_size = 8                # Windows-safe batch size
self.input_size = (224, 224)       # Image size
self.validation_split = 0.2        # Validation split
```

### Model Architecture
- **Base Model**: ResNet50V2 (pre-trained on ImageNet)
- **Classifier**: Dense layers with dropout
- **Output**: 3 classes (benign, malignant, normal)
- **Activation**: Softmax for multi-class classification

## 📈 Expected Output

### Console Output
```
================================================================================
🏥 Breast Cancer Classification - Complete Pipeline
================================================================================
🔧 Configuring TensorFlow for Windows...
✅ TensorFlow version: 2.x.x

🔍 Checking for existing model...
📂 Loading existing model: breast_cancer_master_model.h5
✅ Model loaded successfully!

🔮 Predicting on 5 sample images per class...
📄 benign_001.jpg: benign → benign (0.945)
📄 benign_002.jpg: benign → benign (0.892)
📄 malignant_001.jpg: malignant → malignant (0.978)
📄 malignant_002.jpg: malignant → malignant (0.934)
📄 normal_001.jpg: normal → normal (0.967)

📊 Evaluating model performance...
📈 Model Performance:
   Accuracy: 0.9333 (93.33%)

📋 Classification Report:
              precision    recall  f1-score   support
      benign       0.92      0.95      0.93         5
   malignant       0.96      0.94      0.95         5
      normal       0.93      0.91      0.92         5

📊 Plotting confusion matrix...
📁 Confusion matrix saved as: confusion_matrix_20231201_143022.png

💾 Saving results...
📁 Predictions saved as: predictions_20231201_143022.csv
📁 Confusion matrix saved as: confusion_matrix_20231201_143022.csv
📁 Classification report saved as: classification_report_20231201_143022.txt
📁 Summary saved as: summary_20231201_143022.txt

================================================================================
🎉 PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
📊 Final Accuracy: 0.9333 (93.33%)
📁 Model saved: breast_cancer_master_model.h5
📊 Predictions made: 15
📁 Results saved with timestamp
```

### Generated Files
```
project_folder/
├── breast_cancer_master_model.h5          # Trained model
├── predictions_20231201_143022.csv        # Detailed predictions
├── confusion_matrix_20231201_143022.csv   # Confusion matrix data
├── confusion_matrix_20231201_143022.png   # Confusion matrix plot
├── classification_report_20231201_143022.txt  # Detailed report
├── summary_20231201_143022.txt            # Summary statistics
└── training_history_20231201_143022.png   # Training plots (if trained)
```

## 🏥 Model Architecture

### Transfer Learning Setup
```python
# Base model (frozen)
base_model = ResNet50V2(weights='imagenet', include_top=False)

# Custom classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes
])
```

### Training Configuration
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

## 🔍 Detailed Function Breakdown

### Core Functions

#### 1. `configure_tensorflow()`
- Configures TensorFlow for Windows compatibility
- Sets GPU memory growth
- Disables multiprocessing
- Configures thread settings

#### 2. `verify_dataset_structure()`
- Checks for required class folders
- Counts images in each class
- Validates file formats
- Reports dataset statistics

#### 3. `create_data_generators()`
- Creates training and validation generators
- Applies data augmentation for training
- Uses Windows-safe settings (workers=0, use_multiprocessing=False)

#### 4. `build_model()`
- Loads pre-trained ResNet50V2
- Adds custom classifier layers
- Compiles model with appropriate settings

#### 5. `train_model()`
- Trains model with callbacks
- Uses Windows-safe training settings
- Saves best model based on validation accuracy

#### 6. `predict_sample_images()`
- Randomly samples images from each class
- Makes predictions with confidence scores
- Returns detailed prediction results

#### 7. `evaluate_model()`
- Calculates accuracy, confusion matrix, classification report
- Plots confusion matrix
- Returns comprehensive evaluation metrics

#### 8. `save_results()`
- Saves all results to various file formats
- Creates timestamped files
- Exports data for further analysis

## 🚨 Windows Compatibility Features

### Critical Settings
```python
# Data generators
workers=0,                    # No multiprocessing
use_multiprocessing=False,    # Disable multiprocessing
max_queue_size=10            # Reduced queue size

# Model training
workers=0,                    # No multiprocessing
use_multiprocessing=False,    # Disable multiprocessing
max_queue_size=10            # Reduced queue size

# Environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

### Performance Optimizations
- **Small batch size**: 8 (Windows-safe)
- **Single-threaded operation**: No multiprocessing
- **Memory management**: GPU memory growth
- **Error handling**: Robust exception handling

## 📊 Performance Metrics

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Visualization
- **Confusion Matrix**: Visual representation of predictions vs true labels
- **Training History**: Accuracy and loss plots over epochs
- **Sample Predictions**: Individual image predictions with confidence

## 🔧 Customization Options

### Modify Training Parameters
```python
# In the class initialization
self.epochs = 50              # More epochs
self.learning_rate = 1e-3     # Higher learning rate
self.batch_size = 16          # Larger batch size (if memory allows)
```

### Change Model Architecture
```python
# In build_model() method
# Use different base model
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# Modify classifier layers
Dense(512, activation='relu'),  # More neurons
Dropout(0.7),                   # Higher dropout
```

### Adjust Data Augmentation
```python
# In create_data_generators() method
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,          # More rotation
    width_shift_range=0.2,      # More shift
    height_shift_range=0.2,
    zoom_range=0.2,             # More zoom
    horizontal_flip=True,
    vertical_flip=True,         # Add vertical flip
    fill_mode='nearest',
    validation_split=self.validation_split
)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **"Dataset path not found"**
```bash
# Check if dataset exists
dir "E:\rudra\project\dataset"

# Verify folder structure
dir "E:\rudra\project\dataset\benign"
dir "E:\rudra\project\dataset\malignant"
dir "E:\rudra\project\dataset\normal"
```

#### 2. **"No images found"**
- Ensure images are in JPG or PNG format
- Check file permissions
- Verify image files are not corrupted

#### 3. **"Memory error"**
```python
# Reduce batch size
self.batch_size = 4  # Even smaller

# Reduce image size
self.input_size = (128, 128)  # Smaller images
```

#### 4. **"Training freezes"**
- Script already uses Windows-safe settings
- Close other applications to free memory
- Restart Python if issues persist

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Dataset path incorrect | Check path and folder structure |
| `MemoryError` | Insufficient memory | Reduce batch size or image size |
| `InvalidDicomError` | Wrong file format | Ensure images are JPG/PNG |
| `PermissionError` | No write permission | Check folder permissions |

## 📈 Performance Tips

### For Better Accuracy
1. **More Data**: Add more images to each class
2. **Data Augmentation**: Increase augmentation parameters
3. **Model Fine-tuning**: Unfreeze base model layers
4. **Hyperparameter Tuning**: Adjust learning rate, batch size

### For Faster Training
1. **Reduce Image Size**: Use smaller input size
2. **Reduce Epochs**: Use fewer training epochs
3. **Use GPU**: Ensure GPU is available and configured
4. **Reduce Batch Size**: If memory is limited

### For Windows Stability
1. **Close Applications**: Free up memory
2. **Use SSD**: Faster I/O for large datasets
3. **Restart Python**: If issues persist
4. **Monitor Memory**: Use Task Manager to monitor usage

## 🎯 Integration Examples

### Basic Usage
```python
from master_breast_cancer_classifier import BreastCancerMasterClassifier

# Create classifier
classifier = BreastCancerMasterClassifier()

# Run complete pipeline
success = classifier.run_complete_pipeline(num_samples=5)
```

### Custom Dataset
```python
# Use different dataset path
classifier = BreastCancerMasterClassifier(
    dataset_path="D:\\my_mammogram_dataset",
    model_path="my_custom_model.h5"
)

# Run with more samples
success = classifier.run_complete_pipeline(num_samples=10)
```

### Batch Processing
```python
# Process multiple datasets
datasets = [
    "E:\\dataset1",
    "E:\\dataset2", 
    "E:\\dataset3"
]

for dataset in datasets:
    classifier = BreastCancerMasterClassifier(dataset_path=dataset)
    classifier.run_complete_pipeline()
```

## 📞 Support

### Getting Help
1. **Check Console Output**: Review detailed error messages
2. **Verify Dataset**: Ensure correct folder structure
3. **Check Dependencies**: Ensure all packages are installed
4. **Monitor Resources**: Check memory and disk space

### Common Commands
```bash
# Check if script works
python master_breast_cancer_classifier.py --help

# Test with minimal setup
python -c "from master_breast_cancer_classifier import BreastCancerMasterClassifier; print('Script loaded successfully')"

# Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

## 🎉 Success!

Once the script completes successfully:
1. **Check Generated Files**: Review all saved results
2. **Analyze Performance**: Review accuracy and confusion matrix
3. **Examine Predictions**: Check individual image predictions
4. **Use Model**: The trained model is ready for new predictions

The master script provides a complete, production-ready breast cancer classification pipeline optimized for Windows environments! 