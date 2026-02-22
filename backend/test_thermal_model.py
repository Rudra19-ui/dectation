#!/usr/bin/env python3
"""Test script to verify thermal model predictions"""

import os
import torch
from PIL import Image
from torchvision import transforms

# Load the model
model_path = "thermal_model.pt"
print(f"Loading model from: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

# Create model architecture
from torchvision import models
import torch.nn as nn

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

# Load weights
checkpoint = torch.load(model_path, map_location='cpu')
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded from model_state_dict")
else:
    model.load_state_dict(checkpoint)
    print("Loaded directly")

model.eval()

# Test transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['healthy', 'sick']

# Test on sample images
print("\n" + "="*50)
print("Testing on sample images from dataset")
print("="*50)

# Test healthy images
healthy_dir = "../thermal data/healthy"
if os.path.exists(healthy_dir):
    healthy_images = [f for f in os.listdir(healthy_dir) if f.endswith('.jpg')][:5]
    print(f"\nTesting {len(healthy_images)} healthy images:")
    for img_name in healthy_images:
        img_path = os.path.join(healthy_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = int(torch.argmax(probs).item())
        
        print(f"  {img_name}: {class_names[idx]} (healthy: {probs[0]:.4f}, sick: {probs[1]:.4f})")

# Test sick images
sick_dir = "../thermal data/sick"
if os.path.exists(sick_dir):
    sick_images = [f for f in os.listdir(sick_dir) if f.endswith('.jpg')][:5]
    print(f"\nTesting {len(sick_images)} sick images:")
    for img_name in sick_images:
        img_path = os.path.join(sick_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = int(torch.argmax(probs).item())
        
        print(f"  {img_name}: {class_names[idx]} (healthy: {probs[0]:.4f}, sick: {probs[1]:.4f})")

print("\nDone!")
