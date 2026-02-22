#!/usr/bin/env python3
"""
Comprehensive Training Script for Breast Cancer Detection
Updated for folder-based split_dataset (train/val/test)
"""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Configuration
DATA_DIR = 'split_dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_TYPE = 'resnet50'  # or 'efficientnet'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = f'models/breast_cancer_detector_{MODEL_TYPE}.pt'

# Data transforms
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

# Datasets and loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model
if MODEL_TYPE == 'resnet50':
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
else:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
model = model.to(DEVICE)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training loop with early stopping
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
early_stop_counter = 0
patience = 7  # Stop if no improvement for 7 epochs

print(f"\n🚀 Training {MODEL_TYPE} for {EPOCHS} epochs on {DEVICE}")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  [Train] Batch {batch_idx+1}/{len(train_loader)}", end='\r')
    train_loss = running_loss / total
    train_acc = correct / total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    # Validate
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  [Val] Batch {batch_idx+1}/{len(val_loader)}", end='\r')
    val_loss = running_loss / total
    val_acc = correct / total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    # Early stopping
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  💾 Model saved! (Best val_loss: {best_val_loss:.4f})")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("  🛑 Early stopping triggered!")
            break

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plot_filename = f'training_history_{MODEL_TYPE}_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
plt.savefig(plot_filename, dpi=300)
print(f"✅ Training curves saved as: {plot_filename}")

# Test on test set
test_model(model, test_loader, DEVICE, MODEL_TYPE)

def test_model(model, test_loader, device, model_type):
    """Test the trained model on test set"""
    print(f"\n🧪 Testing Model on Test Set...")
    
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate test metrics
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    
    print(f"📊 Test Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix(y_true, y_pred, model_type)

def create_training_plots(history, model_type):
    """Create training history plots"""
    print(f"\n📈 Creating Training Plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1 Score plot
    ax3.plot(history['train_f1'], label='Train F1', color='blue')
    ax3.plot(history['val_f1'], label='Validation F1', color='red')
    ax3.set_title('Training and Validation F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True)
    
    # Learning rate plot (if available)
    ax4.text(0.5, 0.5, 'Training Complete!', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    ax4.set_title('Training Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plot_filename = f'training_history_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved as: {plot_filename}")

def create_confusion_matrix(y_true, y_pred, model_type):
    """Create confusion matrix plot"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Normal', 'Benign', 'Malignant']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    cm_filename = f'confusion_matrix_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved as: {cm_filename}")

def main():
    """Main training function"""
    print("🏥 Breast Cancer Detection Model Training")
    print("=" * 60)
    
    # Training parameters
    MODEL_TYPE = 'resnet50'  # Options: 'baseline', 'resnet50', 'vgg16', 'efficientnet_b0'
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Start training
    model, history = train_model(
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    print("\n🎉 Training Complete!")
    print("📋 Next Steps:")
    print("1. Test the web app: python -m streamlit run webapp/streamlit_app.py")
    print("2. Test the API: python api/app.py")
    print("3. Use the trained model for predictions")

if __name__ == "__main__":
    main() 