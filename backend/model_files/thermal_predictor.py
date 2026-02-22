import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class ThermalImagePredictor:
    """Predictor for thermal breast cancer images"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["healthy", "sick"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Default model path - check multiple locations
        if model_path is None:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(current_dir)
            
            possible_paths = [
                os.path.join(backend_dir, "thermal_model.pt"),  # backend/thermal_model.pt
                "thermal_model.pt",  # Current working directory
                os.path.join(backend_dir, "models", "thermal_model.pt"),  # backend/models/thermal_model.pt
            ]
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    model_path = abs_path
                    print(f"Found thermal model at: {model_path}")
                    break
            
            if model_path is None:
                print("Warning: Thermal model not found in any of the expected locations:")
                for path in possible_paths:
                    print(f"  - {os.path.abspath(path)}")
        
        self.model_path = model_path
        self.model = self._load_model()
    
    def _create_model(self):
        """Create the model architecture"""
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
            nn.Linear(256, 2)  # 2 classes: healthy, sick
        )
        
        return model
    
    def _load_model(self):
        """Load the trained model"""
        model = self._create_model()
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Thermal model loaded from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load thermal model: {e}")
                print("Using untrained model - predictions will be random")
        else:
            print("Warning: No thermal model found. Using untrained model.")
            print(f"Searched paths include: {self.model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_bytes: bytes):
        """Preprocess image for prediction"""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_image(self, image_bytes: bytes):
        """Make prediction on thermal image"""
        tensor = self.preprocess_image(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        
        # Debug logging
        print(f"Thermal prediction - Raw probabilities: healthy={probs[0]:.4f}, sick={probs[1]:.4f}")
        print(f"Thermal prediction - Predicted class: {self.class_names[idx]} with confidence {conf:.4f}")
        
        return {
            "prediction": self.class_names[idx],
            "confidence": conf,
            "probabilities": [float(p) for p in probs.cpu().numpy()],
            "classes": self.class_names,
            "scan_type": "thermal"
        }
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_type": "ResNet50",
            "classes": self.class_names,
            "device": str(self.device),
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "scan_type": "thermal"
        }
