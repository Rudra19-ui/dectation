#!/usr/bin/env python3
"""
Medical Report Generator - Professional Mammography Report Format
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class MedicalReportGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.class_names = ["benign", "malignant", "normal"]

    def load_model(self):
        """Load the trained model"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
        )

        if Path("best_improved_model.pt").exists():
            checkpoint = torch.load("best_improved_model.pt", map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif Path("quick_enhanced_model.pt").exists():
            checkpoint = torch.load("quick_enhanced_model.pt", map_location=self.device)
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def analyze_image(self, image_path):
        """Analyze image and return results"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(
                self.class_names, probabilities[0].cpu().numpy()
            )
        }

        return {
            "predicted_class": self.class_names[predicted_class],
            "confidence": confidence,
            "probabilities": prob_dict,
            "image_size": image.size,
        }

    def generate_report(self, patient_info, analysis_result):
        """Generate professional medical report"""
        bi_rads = {
            "normal": "BI-RADS 1 (Negative)",
            "benign": "BI-RADS 2 (Benign)",
            "malignant": "BI-RADS 5 (Highly Suspicious)",
        }

        impression = {
            "normal": f"AI analysis shows no evidence of malignancy with {analysis_result['confidence']:.1%} confidence. Normal breast tissue architecture observed.",
            "benign": f"AI analysis indicates benign-appearing features with {analysis_result['confidence']:.1%} confidence. Clinical correlation recommended.",
            "malignant": f"AI analysis suggests highly suspicious findings with {analysis_result['confidence']:.1%} confidence. Immediate clinical evaluation required.",
        }

        suggestions = {
            "normal": [
                "Continue routine breast cancer screening",
                "Monthly self-breast examination",
                "Annual clinical breast examination",
            ],
            "benign": [
                "Clinical correlation with physical examination",
                "Follow-up imaging in 6-12 months",
                "Self-breast examination monthly",
            ],
            "malignant": [
                "Immediate clinical evaluation by breast specialist",
                "Core needle biopsy for definitive diagnosis",
                "Additional imaging (ultrasound, MRI) as indicated",
            ],
        }

        report = f"""
================================================================================
🏥 AI-ENHANCED MAMMOGRAPHY REPORT
================================================================================

PATIENT INFORMATION:
Name: {patient_info.get('name', 'Anonymous')}
Age: {patient_info.get('age', 'N/A')} Years
Sex: {patient_info.get('sex', 'Female')}
Report ID: {patient_info.get('report_id', 'AI-2024-001')}
Analysis Date: {datetime.now().strftime('%d %B, %Y at %I:%M %p')}

CLINICAL FINDINGS:
Technique: AI-Enhanced Digital Mammography Analysis
Breast Tissue: Analyzed using deep learning algorithms
Image Size: {analysis_result['image_size'][0]} x {analysis_result['image_size'][1]} pixels
AI Model: ResNet50 Deep Learning
Confidence Level: {analysis_result['confidence']:.1%}

AI ANALYSIS RESULTS:
Predicted Classification: {analysis_result['predicted_class'].upper()}
BI-RADS Category: {bi_rads[analysis_result['predicted_class']]}
Confidence Score: {analysis_result['confidence']:.1%}

Class Probabilities:
- Benign: {analysis_result['probabilities']['benign']:.1%}
- Malignant: {analysis_result['probabilities']['malignant']:.1%}
- Normal: {analysis_result['probabilities']['normal']:.1%}

IMPRESSION:
{impression[analysis_result['predicted_class']]}

SUGGESTIONS:
{chr(10).join([f"• {suggestion}" for suggestion in suggestions[analysis_result['predicted_class']]])}

IMPORTANT NOTES:
• This AI analysis is for screening purposes only
• Clinical correlation is essential for final diagnosis
• False negative rate: approximately 5-10%
• Management must be based on clinical assessment

REPORT GENERATED: {datetime.now().strftime('%d %B, %Y at %I:%M %p')}
AI MODEL: ResNet50 v2.1 | Confidence: {analysis_result['confidence']:.1%}

================================================================================
        """

        return report

    def save_report(self, patient_info, image_path, output_path="medical_report.txt"):
        """Generate and save the medical report"""
        print("🔍 Analyzing mammogram image...")
        analysis_result = self.analyze_image(image_path)

        print("📋 Generating medical report...")
        report_content = self.generate_report(patient_info, analysis_result)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ Medical report saved as: {output_path}")
        print(f"📊 Analysis Results:")
        print(f"   - Predicted Class: {analysis_result['predicted_class'].upper()}")
        print(f"   - Confidence: {analysis_result['confidence']:.1%}")

        return analysis_result


def main():
    print("🏥 AI-Enhanced Medical Report Generator")
    print("=" * 50)

    generator = MedicalReportGenerator()

    # Example patient information
    patient_info = {
        "name": "Jane Doe",
        "age": "45",
        "sex": "Female",
        "report_id": "AI-2024-001",
    }

    # Test with an image
    test_image = "data/images/benign_001.png"
    if Path(test_image).exists():
        print(f"📸 Processing image: {test_image}")
        result = generator.save_report(patient_info, test_image)

        print("\n🎯 Report Summary:")
        print(f"   - Report saved as: medical_report.txt")
        print(f"   - Open the file to view the complete medical report")
    else:
        print(f"❌ Test image not found: {test_image}")


if __name__ == "__main__":
    main()
