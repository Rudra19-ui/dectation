"""Test script to verify model predictions are consistent."""
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Load model
from models.transfer_learning import get_transfer_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'breast_cancer_detector_resnet50.pt')
print('Loading model from:', MODEL_PATH)
print('Model exists:', os.path.exists(MODEL_PATH))

model, _ = get_transfer_model('resnet50', num_classes=3, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print('Model loaded and set to eval mode')

# Check if model has dropout and if it's disabled
print('\nModel FC layer structure:')
print(model.fc)

# Create a test image with random pixels
np.random.seed(42)
test_img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
test_img = Image.fromarray(test_img_array)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
input_tensor = transform(test_img).unsqueeze(0)

# Run prediction 5 times
print('\nRunning prediction 5 times with same random image:')
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
for i in range(5):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        print(f'  Run {i+1}: {CLASS_NAMES[pred.item()]} ({conf.item()*100:.2f}%)')

# Test with a real image if available
test_image_path = os.path.join(os.path.dirname(__file__), 'reports', 'uploaded_image.png')
if os.path.exists(test_image_path):
    print(f'\nTesting with real image: {test_image_path}')
    real_img = Image.open(test_image_path).convert('RGB')
    input_tensor2 = transform(real_img).unsqueeze(0)
    
    print('Running prediction 5 times with real image:')
    for i in range(5):
        with torch.no_grad():
            output = model(input_tensor2)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            print(f'  Run {i+1}: {CLASS_NAMES[pred.item()]} ({conf.item()*100:.2f}%)')
else:
    print(f'\nTest image not found at {test_image_path}')

print('\nTest complete!')
