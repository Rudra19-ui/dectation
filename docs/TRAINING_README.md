# Breast Cancer Classification Training

This directory contains scripts for training a CNN model to classify breast cancer mammogram images into three classes: **benign**, **malignant**, and **normal**.

## 📁 Files

- `train_breast_cancer_cnn.py` - Main training script
- `predict_breast_cancer.py` - Prediction script for trained model
- `verify_dataset.py` - Dataset verification script
- `TRAINING_README.md` - This file

## 🚀 Quick Start

### 1. Verify Your Dataset

First, ensure your dataset is properly structured:

```bash
python verify_dataset.py
```

Your dataset should have this structure:
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

### 2. Train the Model

```bash
python train_breast_cancer_cnn.py
```

### 3. Make Predictions

```bash
python predict_breast_cancer.py
```

## 🔧 Configuration

### Training Parameters

The training script uses these default parameters:

- **Input Size**: 224x224 pixels
- **Batch Size**: 16 (optimized for Windows)
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 1e-4
- **Validation Split**: 20%
- **Model**: ResNet50V2 with transfer learning

### Model Architecture

```
Input (224x224x3)
    ↓
ResNet50V2 (frozen)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (512, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (3, Softmax) → Output
```

## 📊 Features

### Training Features

- ✅ **Transfer Learning**: Uses pre-trained ResNet50V2
- ✅ **Data Augmentation**: Rotation, zoom, flip, shear
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Learning Rate Scheduling**: Reduces LR when plateauing
- ✅ **Model Checkpointing**: Saves best model
- ✅ **Windows Compatible**: No multiprocessing issues
- ✅ **Validation Split**: Automatic train/validation split

### Evaluation Features

- ✅ **Training History Plots**: Loss, accuracy, precision, recall
- ✅ **Confusion Matrix**: Visual classification results
- ✅ **Classification Report**: Detailed metrics
- ✅ **Model Summary**: Architecture overview

### Prediction Features

- ✅ **Single Image Prediction**: Predict one image at a time
- ✅ **Batch Prediction**: Predict multiple images
- ✅ **Interactive Mode**: Command-line interface
- ✅ **Visualization**: Results with plots
- ✅ **Confidence Scores**: Probability for each class

## 🎯 Expected Output

### Training Output

```
============================================================
🏥 Breast Cancer Classification Training
============================================================
📁 Dataset path: E:\rudra\project\dataset
🖼️ Input size: (224, 224)
📦 Batch size: 16
🔄 Epochs: 50
📊 Classes: ['benign', 'malignant', 'normal']
============================================================

✅ benign: 150 images
✅ malignant: 120 images
✅ normal: 100 images

📂 Creating data generators...
✅ Data generators created!
📊 Training samples: 296
📊 Validation samples: 74
📊 Classes: {'benign': 0, 'malignant': 1, 'normal': 2}

🔧 Creating CNN model...
✅ Model created successfully!

🚀 Starting training...
⏱️ Training for 50 epochs...

Epoch 1/50
19/19 [==============================] - 45s 2s/step - loss: 1.0986 - accuracy: 0.3333 - precision: 0.3333 - recall: 0.3333 - val_loss: 1.0986 - val_accuracy: 0.3333 - val_precision: 0.3333 - val_recall: 0.3333

...

📊 Training history saved as: training_history_20250802_1430.png
📊 Confusion matrix saved as: confusion_matrix_20250802_1430.png

💾 Model saved as: breast_cancer_model.h5

============================================================
🎉 Training Complete!
============================================================
📁 Model saved: breast_cancer_model.h5
📊 Best validation accuracy: 0.8500
📊 Best validation loss: 0.3200
```

### Prediction Output

```
============================================================
🔍 Breast Cancer Classification Prediction
============================================================
✅ Model loaded successfully from breast_cancer_model.h5

🔍 Predicting for: test_image.jpg

📊 Prediction Results:
  Image: test_image.jpg
  Predicted Class: benign
  Confidence: 0.8500 (85.00%)

📊 Class Probabilities:
  benign: 0.8500 (85.00%)
  malignant: 0.1000 (10.00%)
  normal: 0.0500 (5.00%)

📊 Prediction visualization saved as: prediction_test_image.png
```

## 🛠️ Usage Examples

### Training

```bash
# Basic training
python train_breast_cancer_cnn.py

# Check dataset first
python verify_dataset.py
```

### Prediction

```bash
# Interactive mode
python predict_breast_cancer.py

# Single image prediction
python predict_breast_cancer.py --image path/to/image.jpg

# Batch prediction
python predict_breast_cancer.py --folder path/to/images/
```

### Dataset Verification

```bash
# Check dataset structure
python verify_dataset.py

# Create dataset structure if missing
python verify_dataset.py --create
```

## 📈 Performance Metrics

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   ❌ Dataset path not found: E:\rudra\project\dataset
   ```
   **Solution**: Create the dataset folder with proper structure

2. **Model Not Found**
   ```
   ❌ Model not found: breast_cancer_model.h5
   ```
   **Solution**: Train the model first using `train_breast_cancer_cnn.py`

3. **Memory Issues**
   ```
   Out of memory error
   ```
   **Solution**: Reduce batch size in the script (change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`)

4. **Slow Training**
   ```
   Training is very slow
   ```
   **Solution**: 
   - Use GPU if available
   - Reduce image size
   - Use fewer epochs

### Performance Tips

1. **For Better Accuracy**:
   - Use more training data (100+ images per class)
   - Ensure balanced classes
   - Use high-quality images
   - Try different architectures

2. **For Faster Training**:
   - Use GPU acceleration
   - Reduce image size
   - Use smaller batch size
   - Reduce epochs

3. **For Windows Compatibility**:
   - Script already optimized for Windows
   - No multiprocessing used
   - Small batch size (16)

## 📁 Output Files

After training, you'll get these files:

- `breast_cancer_model.h5` - Trained model
- `best_model.h5` - Best model during training
- `training_history_YYYYMMDD_HHMM.png` - Training plots
- `confusion_matrix_YYYYMMDD_HHMM.png` - Confusion matrix
- `prediction_*.png` - Prediction visualizations

## 🚀 Next Steps

After successful training:

1. **Test the model** on new images
2. **Fine-tune hyperparameters** if needed
3. **Deploy the model** for production use
4. **Monitor performance** on real-world data
5. **Retrain periodically** with new data

## 📞 Support

If you encounter issues:

1. **Check dataset structure** with `verify_dataset.py`
2. **Ensure all dependencies** are installed
3. **Check file paths** are correct
4. **Review error messages** carefully
5. **Try with smaller dataset** first

## 🎉 Success!

Once training is complete, you'll have a trained CNN model that can classify breast cancer mammogram images with high accuracy. The model can be used for:

- Medical image analysis
- Research purposes
- Educational demonstrations
- Further development and improvement 