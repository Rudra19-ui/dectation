#!/usr/bin/env python3
"""
Retraining Script for Breast Cancer Detection with Class Balance
This script addresses the model bias issue by using class weights and balanced sampling.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import numpy as np
from datetime import datetime

# Configuration
DATA_DIR = 'datasets/split_dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'models/breast_cancer_detector_resnet50.pt'

CLASS_NAMES = ['Normal', 'Benign', 'Malignant']

print(f"Using device: {DEVICE}")
print(f"Training data: {TRAIN_DIR}")

# Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)} images")
print(f"  Val: {len(val_dataset)} images")
print(f"  Test: {len(test_dataset)} images")

# Calculate class weights for balanced training
class_counts = np.bincount(train_dataset.targets)
print(f"\nClass distribution in training set:")
for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {name}: {count} images ({100*count/len(train_dataset):.1f}%)")

# Calculate weights (inverse frequency)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
print(f"\nClass weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")

# Create weighted sampler for balanced training
sample_weights = [class_weights[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Create model with same architecture as inference
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom FC layer matching the inference architecture
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    
    return model

model = create_model()
model = model.to(DEVICE)

# Print model FC architecture
print(f"\nModel FC architecture:")
print(model.fc)

# Loss with class weights
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer - only train FC layers initially
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}, acc={100.*correct/total:.2f}%")
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    
    print("\nPer-class validation accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            print(f"  {name}: {100.*class_correct[i]/class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return val_loss, val_acc

# Training loop
print(f"\n{'='*60}")
print(f"Starting training for {EPOCHS} epochs")
print(f"{'='*60}")

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"[*] Saved best model with val_acc: {val_acc:.2f}%")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"{'='*60}")

# Test evaluation
print("\nEvaluating on test set...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
print(f"\nTest Accuracy: {test_acc:.2f}%")
