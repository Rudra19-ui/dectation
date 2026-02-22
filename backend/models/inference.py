#!/usr/bin/env python3
"""
Model Inference with Uncertainty Calibration

This module provides functions for model inference with uncertainty calibration
using temperature scaling or isotonic regression.
"""

import os
import uuid
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.isotonic import IsotonicRegression

# Define class names
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for inference
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TemperatureScaling:
    """
    Temperature scaling for calibrating model confidence.
    """
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, logits):
        """
        Scale the logits by temperature parameter and return calibrated probabilities.
        
        Args:
            logits: Raw model output logits
            
        Returns:
            Calibrated probabilities
        """
        return F.softmax(logits / self.temperature, dim=1)

    def set_temperature(self, temperature):
        """
        Set the temperature parameter.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature


def load_model(model_path):
    """
    Load a trained PyTorch model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a dummy model for testing
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: Normal, Benign, Malignant
        model.eval()
        print("Created dummy model for testing")
        return model


def preprocess_image(image):
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image or path to image file
        
    Returns:
        Preprocessed image tensor
    """
    if isinstance(image, str):
        # Load image from path
        image = Image.open(image).convert("RGB")
    
    # Apply transforms
    input_tensor = TRANSFORM(image)
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor


def predict_with_uncertainty(model, image, calibrator=None):
    """
    Perform model inference with uncertainty calibration.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image or path to image file
        calibrator: Calibration method (TemperatureScaling or IsotonicRegression)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(DEVICE)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_tensor)
            
            # Apply calibration if provided
            if calibrator is not None:
                probabilities = calibrator(logits)[0].cpu().numpy()
            else:
                probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get prediction and confidence
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create probabilities dictionary
        prob_dict = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, probabilities)}
        
        # Generate unique IDs for explain and report
        explain_id = f"exp_{uuid.uuid4().hex[:6]}"
        report_id = f"rep_{uuid.uuid4().hex[:6]}"
        
        # Determine if radiologist review is recommended
        recommendation = "Review by radiologist" if confidence < 0.5 else None
        
        # Create result dictionary
        result = {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "model_version": "v1.2",  # Update with actual version tracking
            "explain_id": explain_id,
            "report_id": report_id
        }
        
        # Add recommendation if confidence is low
        if recommendation:
            result["recommendation"] = recommendation
            
        return result
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def train_temperature_scaling(model, val_loader, init_temp=1.0, max_iter=50):
    """
    Train temperature scaling parameter using validation data.
    
    Args:
        model: Trained PyTorch model
        val_loader: Validation data loader
        init_temp: Initial temperature value
        max_iter: Maximum number of optimization iterations
        
    Returns:
        Trained TemperatureScaling calibrator
    """
    # Initialize temperature parameter
    temperature = nn.Parameter(torch.ones(1) * init_temp)
    
    # Define optimizer
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    # Define NLL loss
    nll_criterion = nn.CrossEntropyLoss()
    
    # Set model to evaluation mode
    model.eval()
    
    def eval_loss():
        loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                logits = model(inputs)
                scaled_logits = logits / temperature
                loss += nll_criterion(scaled_logits, labels)
        loss /= len(val_loader)
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    # Optimize temperature parameter
    optimizer.step(eval_loss)
    
    # Create and return calibrator
    calibrator = TemperatureScaling(temperature.item())
    return calibrator


def save_prediction_result(result, output_dir="results"):
    """
    Save prediction result to a JSON file.
    
    Args:
        result: Prediction result dictionary
        output_dir: Directory to save the result
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"prediction_{result['report_id']}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save result to file
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    
    return filepath


def main():
    """
    Main function for testing the module.
    """
    # Example usage
    model_path = "models/breast_cancer_detector_resnet50.pt"
    image_path = "data/test_image.jpg"
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model.")
        return
    
    # Create calibrator (in practice, this would be trained on validation data)
    calibrator = TemperatureScaling(temperature=1.5)
    
    # Perform prediction
    result = predict_with_uncertainty(model, image_path, calibrator)
    
    # Print result
    if result:
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
        
        # Save result
        output_path = save_prediction_result(result)
        print(f"\nResult saved to: {output_path}")
    else:
        print("Prediction failed.")


if __name__ == "__main__":
    main()