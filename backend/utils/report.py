#!/usr/bin/env python3
"""
Medical Report Generation

This module provides functions for generating professional PDF reports
with image, heatmap, metrics, version, and disclaimer.
"""

try:
    from fpdf import FPDF
except ImportError:
    from fpdf2 import FPDF

import os
import uuid
import json
from datetime import datetime


class DiagnosisReport(FPDF):
    """
    PDF report generator for breast cancer diagnosis.
    """
    def __init__(
        self,
        image_path,
        prediction,
        confidence,
        probabilities,
        heatmap_path,
        patient_info=None,
        model_version="v1.2",
        output_dir="reports",
    ):
        super().__init__()
        self.image_path = image_path
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.heatmap_path = heatmap_path
        self.patient_info = patient_info or {}
        self.model_version = model_version
        self.output_dir = output_dir
        self.set_auto_page_break(auto=True, margin=15)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def header(self):
        """
        Define the report header.
        """
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Breast Cancer Diagnostic Report", ln=True, align="C")
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Model Version: {self.model_version}", ln=True, align="C")
        self.cell(
            0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align="C"
        )
        self.ln(5)

    def footer(self):
        """
        Define the report footer.
        """
        self.set_y(-20)
        self.set_font("Arial", "I", 8)
        self.multi_cell(0, 5, "DISCLAIMER: This is an automated analysis and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.")
        self.set_y(-10)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_patient_info(self):
        """
        Add anonymized patient information to the report.
        """
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Patient Information (Anonymized)", ln=True)
        self.set_font("Arial", "", 10)
        
        # Add patient info if available
        if self.patient_info:
            for key, value in self.patient_info.items():
                if key.lower() not in ["name", "full name", "address", "phone", "ssn", "social security"]:
                    self.cell(0, 6, f"{key}: {value}", ln=True)
        else:
            self.cell(0, 6, "Patient ID: Anonymous", ln=True)
            self.cell(0, 6, "Age Group: Not provided", ln=True)
        
        self.ln(5)

    def add_prediction_results(self):
        """
        Add prediction results to the report.
        """
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Diagnostic Results", ln=True)
        self.set_font("Arial", "", 10)
        
        # Prediction and confidence
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, f"Prediction: {self.prediction}", ln=True)
        self.cell(0, 8, f"Confidence: {self.confidence:.1%}", ln=True)
        self.set_font("Arial", "", 10)
        
        # Class probabilities
        self.ln(2)
        self.cell(0, 6, "Class Probabilities:", ln=True)
        for class_name, prob in self.probabilities.items():
            self.cell(0, 6, f"  - {class_name}: {prob:.1%}", ln=True)
        
        self.ln(5)

    def add_recommendation(self):
        """
        Add recommendation based on prediction confidence.
        """
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Recommendation", ln=True)
        self.set_font("Arial", "", 10)
        
        if self.prediction == "Malignant":
            self.multi_cell(0, 6, "IMPORTANT: This result indicates potential malignancy. Immediate consultation with a healthcare professional is strongly recommended.")
        elif self.confidence < 0.5:
            self.multi_cell(0, 6, "Due to low confidence in this prediction, review by a radiologist is recommended.")
        else:
            self.multi_cell(0, 6, "Regular follow-up with your healthcare provider is recommended.")
        
        self.ln(5)

    def generate(self):
        """
        Generate the complete PDF report.
        
        Returns:
            Dictionary containing report_id and report_url
        """
        self.add_page()
        
        # Add patient information
        self.add_patient_info()
        
        # Add prediction results
        self.add_prediction_results()
        
        # Add images
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Diagnostic Images", ln=True)
        
        # Original image
        if os.path.exists(self.image_path):
            self.set_font("Arial", "", 10)
            self.cell(0, 6, "Original Image:", ln=True)
            self.image(self.image_path, x=10, y=None, w=80)
            self.ln(85)  # Space for the image
        
        # Heatmap image
        if os.path.exists(self.heatmap_path):
            self.set_font("Arial", "", 10)
            self.cell(0, 6, "Grad-CAM Heatmap (Areas of Interest):", ln=True)
            self.image(self.heatmap_path, x=10, y=None, w=80)
            self.ln(85)  # Space for the image
        
        # Add recommendation
        self.add_recommendation()
        
        # Generate unique report ID
        report_id = f"rep_{uuid.uuid4().hex[:6]}"
        filename = f"{report_id}.pdf"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save the PDF
        self.output(output_path)
        
        # Return report information
        return {
            "report_id": report_id,
            "report_url": f"/reports/{filename}"
        }
