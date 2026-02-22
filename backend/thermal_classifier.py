#!/usr/bin/env python3
"""
Thermal Image Breast Cancer Classifier
Trains a model to classify breast thermal images as healthy or sick
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ThermalImageDataset(Dataset):
    """Dataset for thermal breast images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with 'healthy' and 'sick' subdirectories
            transform: Optional transform to be applied
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['healthy', 'sick']
        self.class_to_idx = {'healthy': 0, 'sick': 1}
        self.samples = []
        
        # Load all images
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images")
        print(f"  - Healthy: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  - Sick: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


class ThermalClassifier:
    """Thermal image classifier using transfer learning"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_names = ['healthy', 'sick']
    
    def create_model(self):
        """Create ResNet50 model with custom classifier"""
        # Use EfficientNet for better performance
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the classifier
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
            nn.Linear(256, self.num_classes)
        )
        
        # Unfreeze last few layers for fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        self.model = model.to(device)
        return self.model
    
    def train(self, train_loader, val_loader, epochs=30, learning_rate=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW([
            {'params': self.model.layer4.parameters(), 'lr': learning_rate * 0.1},
            {'params': self.model.fc.parameters(), 'lr': learning_rate}
        ], weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        best_model_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = 100.0 * val_correct / val_total
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"  New best model! Val Acc: {val_acc:.2f}%")
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Calculate final metrics
        print("\n" + "="*50)
        print("Final Evaluation on Validation Set:")
        print("="*50)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        
        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"\nWeighted Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return history, {
            'accuracy': best_val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat()
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=device)
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {path}")


def main():
    """Main training function"""
    # Configuration
    data_dir = Path("../thermal data")  # Path to thermal data directory
    model_save_path = "thermal_model.pt"
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    train_split = 0.8
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please ensure the thermal data directory exists with 'healthy' and 'sick' subdirectories")
        return
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = ThermalImageDataset(data_dir, transform=None)
    
    if len(full_dataset) == 0:
        print("Error: No images found in the dataset!")
        return
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use stratified split
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        indices, train_size=train_split, stratify=labels, random_state=42
    )
    
    # Create subsets with appropriate transforms
    class SubsetWithTransform(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            img_path, label = self.dataset.samples[self.indices[idx]]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = SubsetWithTransform(full_dataset, train_indices, 
                                        ThermalClassifier().transform)
    val_dataset = SubsetWithTransform(full_dataset, val_indices, 
                                      ThermalClassifier().val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create and train model
    print("\nCreating model...")
    classifier = ThermalClassifier(num_classes=2)
    classifier.create_model()
    
    print("\nStarting training...")
    history, metrics = classifier.train(train_loader, val_loader, epochs, learning_rate)
    
    # Save model
    classifier.save_model(model_save_path)
    
    # Save training history
    history_path = "thermal_training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'metrics': metrics,
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'train_split': train_split
            }
        }, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\nTraining complete!")
    print(f"Final model accuracy: {metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
