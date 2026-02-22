#!/usr/bin/env python3
"""Test script to verify model predictions"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configuration
MODEL_PATH = 'models/breast_cancer_detector_resnet50.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# IMPORTANT: Class names must match ImageFolder's alphabetical ordering used during training
# ImageFolder assigns labels alphabetically: Benign=0, Malignant=1, Normal=2
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

def create_model():
    """Create model with same architecture as training"""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3),
    )
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def test_model(model_path, test_images):
    """Test model on given images"""
    print(f"Loading model from: {model_path}")
    print(f"Device: {DEVICE}")
    
    # Load model
    model = create_model()
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    print(f"\nModel loaded successfully!")
    print("=" * 60)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\nImage not found: {img_path}")
            continue
            
        print(f"\nTesting: {img_path}")
        
        # Preprocess and predict
        input_tensor = preprocess_image(img_path).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            
        pred_class = CLASS_NAMES[pred.item()]
        conf_score = confidence.item() * 100
        
        print(f"  Prediction: {pred_class}")
        print(f"  Confidence: {conf_score:.2f}%")
        print(f"  All probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0].cpu().numpy())):
            print(f"    {name}: {prob*100:.2f}%")

if __name__ == '__main__':
    # Find some test images from the dataset
    test_dir = 'datasets/split_dataset/test'
    test_images = []
    
    if os.path.exists(test_dir):
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    test_images.append(os.path.join(class_dir, files[0]))
                    test_images.append(os.path.join(class_dir, files[min(5, len(files)-1)]))
    
    # Also check for uploaded images
    upload_dir = 'reports'
    if os.path.exists(upload_dir):
        files = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for f in files[:3]:
            test_images.append(os.path.join(upload_dir, f))
    
    if not test_images:
        print("No test images found!")
        sys.exit(1)
    
    print(f"Found {len(test_images)} test images")
    test_model(MODEL_PATH, test_images)
