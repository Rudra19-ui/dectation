# BatchNorm Debugging Guide

## 🔍 **Why BatchNorm Behaves Differently in Test vs Training**

### **Training Mode (`model.train()`)**
- Uses **current batch statistics** (mean/variance)
- Updates running statistics for future use
- Requires batch size > 1 for stable statistics
- Formula: `y = γ * (x - μ_batch) / √(σ²_batch + ε) + β`

### **Evaluation Mode (`model.eval()`)**
- Uses **pre-computed running statistics** from training
- No updates to running statistics
- Can work with batch size = 1
- Formula: `y = γ * (x - μ_running) / √(σ²_running + ε) + β`

## 🛠️ **Correct PyTorch Inference Setup**

### **1. Always Set Model to Evaluation Mode**
```python
model.eval()  # CRITICAL for BatchNorm layers
```

### **2. Use torch.no_grad() Context**
```python
with torch.no_grad():  # Disable gradient computation
    output = model(input_tensor)
```

### **3. Complete Inference Pattern**
```python
def predict_safely(model, input_tensor):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(input_tensor)
    return output
```

## 🔧 **Common BatchNorm Issues & Solutions**

### **Issue 1: "Expected more than 1 value per channel"**
**Cause**: Batch size = 1 in training mode
**Solution**: 
```python
model.eval()  # Switch to evaluation mode
# OR use larger batch size
```

### **Issue 2: Inconsistent Predictions**
**Cause**: Model switching between train/eval modes
**Solution**:
```python
# Always ensure consistent mode
model.eval()  # Before every inference
with torch.no_grad():
    output = model(input_tensor)
```

### **Issue 3: BatchNorm Statistics Not Updated**
**Cause**: `track_running_stats=False` or model in eval mode
**Solution**:
```python
# For training
model.train()
# For inference
model.eval()
```

## 🔍 **Debugging Techniques**

### **1. Check Model Mode**
```python
print(f"Model training mode: {model.training}")
print(f"Model eval mode: {not model.training}")
```

### **2. Inspect BatchNorm Layers**
```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f"BatchNorm layer: {name}")
        print(f"  track_running_stats: {module.track_running_stats}")
        print(f"  running_mean: {module.running_mean.shape}")
        print(f"  running_var: {module.running_var.shape}")
        print(f"  training: {module.training}")
```

### **3. Test Different Batch Sizes**
```python
for batch_size in [1, 2, 4, 8]:
    try:
        model.eval()
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Batch size {batch_size}: successful")
    except Exception as e:
        print(f"❌ Batch size {batch_size}: failed - {e}")
```

### **4. Compare Train vs Eval Outputs**
```python
# Test training mode
model.train()
with torch.no_grad():
    output_train = model(input_tensor)

# Test evaluation mode
model.eval()
with torch.no_grad():
    output_eval = model(input_tensor)

# Compare (should be different due to BatchNorm)
diff = torch.abs(output_train - output_eval).mean()
print(f"BatchNorm mode difference: {diff.item():.6f}")
```

## 📊 **Best Practices**

### **For Training:**
```python
model.train()  # Set training mode
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
```

### **For Inference:**
```python
model.eval()  # Set evaluation mode
with torch.no_grad():  # Disable gradients
    output = model(input_tensor)
    predictions = torch.argmax(output, dim=1)
```

### **For Model Loading:**
```python
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Always set to eval mode after loading
```

## 🚨 **Common Mistakes**

1. **Forgetting `model.eval()`**: Causes inconsistent predictions
2. **Using `torch.no_grad()` without `model.eval()`**: Still uses batch statistics
3. **Mixing train/eval modes**: Leads to unpredictable behavior
4. **Batch size = 1 in training mode**: Causes BatchNorm errors

## ✅ **Verification Checklist**

- [ ] Model set to `eval()` mode before inference
- [ ] Using `torch.no_grad()` context
- [ ] BatchNorm layers have `track_running_stats=True`
- [ ] Running statistics are properly initialized
- [ ] Consistent mode throughout inference pipeline

## 🎯 **Key Takeaways**

1. **Always use `model.eval()` for inference**
2. **Always use `torch.no_grad()` for inference**
3. **BatchNorm needs running statistics for evaluation**
4. **Test with different batch sizes to ensure stability**
5. **Check model mode and BatchNorm configuration**

This guide ensures your models work correctly in both training and inference modes! 