import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class BreastCancerPredictor:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["benign", "malignant", "normal"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model_path = model_path or ("best_improved_model.pt" if os.path.exists("best_improved_model.pt") else "quick_enhanced_model.pt")
        self.model = self._load_model()

    def _load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
        )
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict_image(self, image_bytes: bytes):
        tensor = self.preprocess_image(image_bytes)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        return {
            "prediction": self.class_names[idx],
            "confidence": conf,
            "probabilities": [float(p) for p in probs.cpu().numpy()],
            "classes": self.class_names,
        }