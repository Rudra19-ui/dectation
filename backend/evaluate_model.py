#!/usr/bin/env python3
"""Evaluate model accuracy on test dataset"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

# Configuration
MODEL_PATH = 'models/breast_cancer_detector_resnet50.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']  # Alphabetical order used by ImageFolder
TEST_DIR = 'datasets/split_dataset/test'

def create_model():
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

def evaluate_model():
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Load model
    model = create_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluate each class
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nTesting {class_name}: {len(files)} images")
        
        for filename in files:
            filepath = os.path.join(class_dir, filename)
            try:
                image = Image.open(filepath).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    confidence, pred = torch.max(probabilities, 0)
                
                pred_class = CLASS_NAMES[pred.item()]
                conf_score = confidence.item() * 100
                
                results[class_name]['total'] += 1
                results[class_name]['confidences'].append(conf_score)
                
                if pred_class == class_name:
                    results[class_name]['correct'] += 1
                else:
                    # Show misclassified examples
                    print(f"  WRONG: {filename} -> Predicted: {pred_class} ({conf_score:.1f}%)")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_correct = 0
    total_samples = 0
    
    for class_name in CLASS_NAMES:
        r = results[class_name]
        if r['total'] > 0:
            accuracy = 100 * r['correct'] / r['total']
            avg_conf = sum(r['confidences']) / len(r['confidences'])
            total_correct += r['correct']
            total_samples += r['total']
            print(f"{class_name}: {r['correct']}/{r['total']} correct ({accuracy:.1f}%), avg confidence: {avg_conf:.1f}%")
    
    overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")

if __name__ == '__main__':
    evaluate_model()
