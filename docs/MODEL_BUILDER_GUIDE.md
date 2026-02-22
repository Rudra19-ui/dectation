# 🤖 Deep Learning Model Builder Guide

## 🎯 Overview
This guide helps you build CNN models using transfer learning for mammogram classification:
- **EfficientNetB0** - Efficient and accurate
- **ResNet50** - Proven architecture with good performance
- **Input**: 224x224x3 images
- **Output**: 3 classes (normal, benign, malignant)
- **Activation**: Softmax
- **Optimizer**: Adam with categorical crossentropy
- **Framework**: PyTorch (primary) and TensorFlow (optional)

## 🚀 Quick Start

### 1. Build ResNet50 Model (PyTorch)
```bash
python build_model.py --model-type resnet50 --save-model
```

### 2. Build EfficientNet Model (PyTorch)
```bash
python build_model.py --model-type efficientnet --save-model
```

### 3. Build with TensorFlow (if available)
```bash
python build_model.py --framework tensorflow --model-type resnet50 --save-model
```

## 📊 Model Architectures

### ResNet50 Architecture
```
Input: 224x224x3
├── ResNet50 Backbone (frozen)
│   ├── Conv1: 7x7, 64 filters
│   ├── MaxPool: 3x3
│   ├── Layer1: 3 blocks, 256 filters
│   ├── Layer2: 4 blocks, 512 filters
│   ├── Layer3: 6 blocks, 1024 filters
│   └── Layer4: 3 blocks, 2048 filters
├── Global Average Pooling
├── Dropout (0.5)
├── Dense: 512 units + ReLU
├── Dropout (0.3)
└── Output: 3 classes (softmax)
```

### EfficientNet-B0 Architecture
```
Input: 224x224x3
├── EfficientNet-B0 Backbone (frozen)
│   ├── Initial Conv: 3x3, 32 filters
│   ├── MBConv1: 16 filters
│   ├── MBConv6: 24 filters
│   ├── MBConv6: 40 filters
│   ├── MBConv6: 80 filters
│   ├── MBConv6: 112 filters
│   ├── MBConv6: 192 filters
│   └── MBConv6: 320 filters
├── Global Average Pooling
├── Dropout (0.5)
├── Dense: 512 units + ReLU
├── Dropout (0.3)
└── Output: 3 classes (softmax)
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--framework` | Framework (pytorch/tensorflow) | `pytorch` |
| `--model-type` | Model type (resnet50/efficientnet) | `resnet50` |
| `--num-classes` | Number of output classes | `3` |
| `--input-size` | Input image size | `224` |
| `--freeze-backbone` | Freeze pretrained layers | `True` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--train-dir` | Training data directory | `split_dataset/train` |
| `--val-dir` | Validation data directory | `split_dataset/val` |
| `--test-dir` | Test data directory | `split_dataset/test` |
| `--batch-size` | Batch size | `32` |
| `--output-dir` | Output directory | `models` |
| `--save-model` | Save model after building | `False` |

## 🎯 Use Cases

### 1. Standard ResNet50 Model
```bash
python build_model.py --model-type resnet50 --save-model
```

### 2. EfficientNet with Custom Learning Rate
```bash
python build_model.py --model-type efficientnet --learning-rate 5e-5 --save-model
```

### 3. TensorFlow Model
```bash
python build_model.py --framework tensorflow --model-type resnet50 --save-model
```

### 4. Custom Input Size
```bash
python build_model.py --input-size 256 --save-model
```

### 5. Unfrozen Backbone (Fine-tuning)
```bash
python build_model.py --freeze-backbone False --learning-rate 1e-5 --save-model
```

## 📁 Expected Directory Structure

### Input Structure
```
split_dataset/
├── train/
│   ├── normal/      (567 images)
│   ├── benign/      (1,915 images)
│   └── malignant/   (1,735 images)
├── val/
│   ├── normal/      (122 images)
│   ├── benign/      (410 images)
│   └── malignant/   (372 images)
└── test/
    ├── normal/      (122 images)
    ├── benign/      (410 images)
    └── malignant/   (372 images)
```

### Output Structure
```
models/
├── pytorch_resnet50_model.pt
├── pytorch_efficientnet_model.pt
├── tensorflow_resnet50_model.h5
└── tensorflow_efficientnet_model.h5

model_summary_pytorch_resnet50.txt
model_summary_pytorch_efficientnet.txt
model_summary_tensorflow_resnet50.txt
model_summary_tensorflow_efficientnet.txt
```

## 🔧 Requirements

### PyTorch Requirements
```bash
pip install torch torchvision torchaudio
pip install pillow numpy pandas matplotlib seaborn tqdm
```

### TensorFlow Requirements (Optional)
```bash
pip install tensorflow
```

## 📊 Model Specifications

### Transfer Learning Setup
- **Pretrained Weights**: ImageNet
- **Backbone Freezing**: Enabled by default
- **Fine-tuning**: Available with `--freeze-backbone False`
- **Data Augmentation**: Built-in for training

### Training Transforms (PyTorch)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Validation Transforms (PyTorch)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Data Augmentation (TensorFlow)
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
```

## 🎯 Model Performance Expectations

### ResNet50
- **Parameters**: ~25M total, ~1M trainable (frozen backbone)
- **Memory**: ~2GB GPU memory
- **Speed**: Fast inference
- **Accuracy**: Good baseline performance

### EfficientNet-B0
- **Parameters**: ~5M total, ~200K trainable (frozen backbone)
- **Memory**: ~1GB GPU memory
- **Speed**: Very fast inference
- **Accuracy**: Competitive with ResNet50

## 📋 Model Summary Example

```
MODEL SUMMARY - PYTORCH
==================================================

Model Type: resnet50
Framework: pytorch
Input Size: 224x224x3
Output Classes: 3
Freeze Backbone: True
Learning Rate: 0.0001
Device: cuda
Total Parameters: 25,557,033
Trainable Parameters: 1,050,371
```

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```bash
   python build_model.py --batch-size 16
   ```

2. **TensorFlow Not Available**
   - Install TensorFlow: `pip install tensorflow`
   - Or use PyTorch: `python build_model.py --framework pytorch`

3. **Dataset Not Found**
   - Check directory structure
   - Ensure images are in PNG/JPG format
   - Verify class directories exist

4. **Model Loading Error**
   - Check file paths
   - Ensure model was saved correctly
   - Verify framework compatibility

### Validation Checks:
- Verifies dataset structure
- Tests model with sample batch
- Checks device availability
- Validates model architecture
- Confirms output shapes

## 🎉 Next Steps

After building the model:

1. **Train the Model**
   ```bash
   python train_model.py --model-path models/pytorch_resnet50_model.pt
   ```

2. **Evaluate Performance**
   ```bash
   python evaluate_model.py --model-path models/pytorch_resnet50_model.pt
   ```

3. **Deploy to Web App**
   ```bash
   python webapp/streamlit_app.py
   ```

## 📊 Model Comparison

| Model | Parameters | Memory | Speed | Accuracy |
|-------|------------|--------|-------|----------|
| ResNet50 | 25M | 2GB | Fast | High |
| EfficientNet-B0 | 5M | 1GB | Very Fast | High |

## 🔄 Model Loading

### PyTorch Model Loading
```python
import torch
from build_model import PyTorchModelBuilder

# Load model
model_builder = PyTorchModelBuilder()
model_builder.load_model("models/pytorch_resnet50_model.pt")

# Use for inference
model_builder.model.eval()
```

### TensorFlow Model Loading
```python
import tensorflow as tf
from build_model import TensorFlowModelBuilder

# Load model
model_builder = TensorFlowModelBuilder()
model_builder.load_model("models/tensorflow_resnet50_model.h5")

# Use for inference
predictions = model_builder.model.predict(images)
```

Your model is now ready for training! 🚀 