# 🔧 Step-by-Step Modifications for improved_training.py

## 🎯 Goal: Reduce Malignant False Negatives from 110 to ~60

### Current Issues in Your Code:
1. **No class weighting** - treats all classes equally
2. **Standard CrossEntropyLoss** - doesn't focus on hard examples
3. **Limited data augmentation** - missing medical-specific augmentations
4. **No malignant-specific evaluation** - focuses only on overall accuracy

---

## 📝 Modification 1: Add Weighted Loss Functions

### A. Add Focal Loss Class (Insert after imports)
```python
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss
```

### B. Modify `train_model` function (Replace lines 105-108)

**OLD CODE:**
```python
def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the model with improved techniques"""
    criterion = nn.CrossEntropyLoss()  # ← REPLACE THIS
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**NEW CODE:**
```python
def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the model with malignant-focused techniques"""
    
    # Calculate class weights (give extra weight to malignant)
    class_weights = torch.FloatTensor([1.0, 2.5, 1.8]).to(device)  # [benign, malignant, normal]
    
    # Use combination of weighted CE and Focal Loss
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_focal = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### C. Modify the loss calculation (Replace lines 125-126)

**OLD CODE:**
```python
loss = criterion(outputs, labels)
```

**NEW CODE:**
```python
# Combined loss: 60% weighted CE + 40% focal loss
loss_ce = criterion_ce(outputs, labels)
loss_focal = criterion_focal(outputs, labels)
loss = 0.6 * loss_ce + 0.4 * loss_focal
```

---

## 📝 Modification 2: Enhanced Data Augmentation

### Replace the `create_transforms()` function (lines 60-78)

**OLD CODE:**
```python
def create_transforms():
    """Create training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

**NEW CODE:**
```python
def create_transforms():
    """Create enhanced training and validation transforms for malignant detection"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=16),
        
        # Enhanced augmentations for medical imaging
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Important for mammograms
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        
        # Enhanced intensity variations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        
        # Add noise and blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Random erasing to force model to focus on different regions
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2))
    ])
```

---

## 📝 Modification 3: Add Malignant-Specific Evaluation

### A. Add malignant metrics function (Insert after `evaluate_model` function)

```python
def calculate_malignant_metrics(all_labels, all_predictions):
    """Calculate specific metrics for malignant class (index 1)"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Malignant-specific metrics (class index 1)
    if len(cm) >= 2:
        malignant_tp = cm[1, 1]
        malignant_fp = cm[0, 1] + (cm[2, 1] if len(cm) > 2 else 0)  # Others predicted as malignant
        malignant_fn = cm[1, 0] + (cm[1, 2] if len(cm) > 2 else 0)  # Malignant predicted as others
        
        malignant_precision = malignant_tp / (malignant_tp + malignant_fp) if (malignant_tp + malignant_fp) > 0 else 0
        malignant_recall = malignant_tp / (malignant_tp + malignant_fn) if (malignant_tp + malignant_fn) > 0 else 0
        malignant_f1 = 2 * (malignant_precision * malignant_recall) / (malignant_precision + malignant_recall) if (malignant_precision + malignant_recall) > 0 else 0
        
        return malignant_precision, malignant_recall, malignant_f1, malignant_fn
    
    return 0, 0, 0, 0
```

### B. Modify the training loop to track malignant metrics (Replace lines 163-179)

**OLD CODE:**
```python
print(f'Epoch {epoch+1}/{num_epochs}:')
print(f'  Train Loss: {train_loss:.4f}')
print(f'  Val Loss: {val_loss:.4f}')
print(f'  Val Accuracy: {val_accuracy:.2f}%')

# Save best model
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': val_accuracy
    }, 'best_improved_model.pt')
    print(f'  ✅ New best model saved! Accuracy: {val_accuracy:.2f}%')
```

**NEW CODE:**
```python
# Calculate malignant-specific metrics during validation
model.eval()
val_predictions = []
val_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

# Get malignant metrics
mal_precision, mal_recall, mal_f1, mal_fn = calculate_malignant_metrics(val_labels, val_predictions)

print(f'Epoch {epoch+1}/{num_epochs}:')
print(f'  Train Loss: {train_loss:.4f}')
print(f'  Val Loss: {val_loss:.4f}')
print(f'  Val Accuracy: {val_accuracy:.2f}%')
print(f'  🎯 Malignant Precision: {mal_precision:.4f}')
print(f'  🎯 Malignant Recall: {mal_recall:.4f}')
print(f'  🎯 Malignant F1: {mal_f1:.4f}')
print(f'  ❌ Malignant False Negatives: {mal_fn}')

# Save best model based on combined score (40% accuracy + 60% malignant F1)
combined_score = 0.4 * (val_accuracy/100) + 0.6 * mal_f1

if combined_score > best_accuracy:
    best_accuracy = combined_score
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': val_accuracy,
        'malignant_f1': mal_f1,
        'malignant_recall': mal_recall,
        'combined_score': combined_score
    }, 'best_malignant_focused_model.pt')
    print(f'  🎯 NEW BEST MODEL! Combined Score: {combined_score:.4f}, Malignant F1: {mal_f1:.4f}')
```

---

## 📝 Modification 4: Enhanced Model Architecture

### Replace the `create_model` function (lines 80-103)

**OLD CODE:**
```python
def create_model(num_classes=3):
    """Create and configure the model"""
    # Use ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model
```

**NEW CODE:**
```python
def create_model(num_classes=3):
    """Create enhanced model for malignant detection"""
    # Use ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers but allow more fine-tuning
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Enhanced classifier architecture
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        # First dense block
        nn.Dropout(0.5),
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
        # Second dense block  
        nn.Dropout(0.35),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        
        # Final classification layer
        nn.Dropout(0.25),
        nn.Linear(512, num_classes)
    )
    
    return model
```

---

## 📝 Modification 5: Add Weighted Random Sampler

### A. Add the weighted sampler function (Insert before `main()`)

```python
def create_weighted_sampler(dataset):
    """Create weighted sampler to balance classes with emphasis on malignant"""
    from torch.utils.data import WeightedRandomSampler
    
    # Count samples per class
    class_counts = [0, 0, 0]  # benign, malignant, normal
    for _, label in dataset:
        class_counts[label] += 1
    
    print(f"📊 Training Class Distribution:")
    class_names = ['benign', 'malignant', 'normal']
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"   {name}: {count} samples")
    
    # Calculate sample weights (higher weight = more likely to be sampled)
    total_samples = sum(class_counts)
    class_weights = []
    for i, count in enumerate(class_counts):
        if i == 1:  # malignant class - give extra weight
            weight = (total_samples / count) * 2.0
        else:
            weight = total_samples / count
        class_weights.append(weight)
    
    print(f"📊 Class Sampling Weights:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"   {name}: {weight:.4f}")
    
    # Create sample weights
    sample_weights = [class_weights[label] for _, label in dataset]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
```

### B. Modify the DataLoader creation in `main()` (Replace line 263)

**OLD CODE:**
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
```

**NEW CODE:**
```python
# Create weighted sampler for balanced training
weighted_sampler = create_weighted_sampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=weighted_sampler, num_workers=0)
```

---

## 📝 Modification 6: Enhanced Final Evaluation

### Replace the final evaluation section in `main()` (Replace lines 286-303)

**OLD CODE:**
```python
accuracy, report, cm = evaluate_model(model, test_loader, device)

print(f"\n🎯 Final Test Results:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"\n📋 Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['benign', 'malignant', 'normal'],
           yticklabels=['benign', 'malignant', 'normal'])
plt.title('Confusion Matrix - Improved Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

**NEW CODE:**
```python
accuracy, report, cm = evaluate_model(model, test_loader, device)

# Get detailed malignant metrics
test_predictions = []
test_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

mal_precision, mal_recall, mal_f1, mal_fn = calculate_malignant_metrics(test_labels, test_predictions)

print(f"\n🎯 Final Test Results:")
print(f"   Overall Accuracy: {accuracy:.4f}")
print(f"   🎯 Malignant Precision: {mal_precision:.4f}")
print(f"   🎯 Malignant Recall: {mal_recall:.4f}")
print(f"   🎯 Malignant F1-Score: {mal_f1:.4f}")
print(f"   ❌ Malignant False Negatives: {mal_fn}")
print(f"\n📋 Classification Report:")
print(report)

# Enhanced confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['benign', 'malignant', 'normal'],
           yticklabels=['benign', 'malignant', 'normal'])
plt.title('Enhanced Confusion Matrix - Malignant Focused Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add malignant error rate
total_malignant = cm[1].sum()
malignant_errors = cm[1, 0] + cm[1, 2]
error_rate = malignant_errors / total_malignant * 100
plt.figtext(0.5, 0.02, 
           f'Malignant Misclassification Rate: {malignant_errors}/{total_malignant} ({error_rate:.1f}%)', 
           ha='center', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('malignant_focused_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🚀 Quick Implementation Steps

### 1. **Backup your current model**
```bash
cp best_improved_model.pt best_improved_model_backup.pt
```

### 2. **Apply all modifications above to `improved_training.py`**

### 3. **Or use the complete enhanced script**
```bash
python malignant_focused_training.py  # The complete solution I provided
```

### 4. **Monitor the key metrics during training:**
- **Malignant F1-Score** (should increase from 0.83 to 0.89+)
- **Malignant Recall** (should increase from 0.82 to 0.90+)
- **False Negatives** (should decrease from 110 to ~60)

### 5. **Expected Training Output:**
```
Epoch 1/40:
  Train Loss: 1.2543
  Val Loss: 0.8934
  Val Accuracy: 78.45%
  🎯 Malignant Precision: 0.8234
  🎯 Malignant Recall: 0.8567
  🎯 Malignant F1: 0.8397
  ❌ Malignant False Negatives: 95
  🎯 NEW BEST MODEL! Combined Score: 0.8178, Malignant F1: 0.8397
```

---

## ⚠️ Important Notes

1. **Training will take longer** due to enhanced augmentations and loss calculations
2. **Monitor malignant metrics** more than overall accuracy
3. **Expect some overall accuracy fluctuation** as we optimize for malignant detection
4. **The model will be more sensitive** to malignant cases (which is good!)
5. **False positives may increase slightly** but false negatives should decrease significantly

These modifications will directly address your malignant classification problem by:
- **Penalizing malignant misclassifications more heavily**
- **Focusing on hard-to-classify examples**
- **Balancing the training data**
- **Using medical-specific augmentations**
- **Tracking malignant-specific metrics**
