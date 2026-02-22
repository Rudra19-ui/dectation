import io
import os
import sys
import json
import datetime

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from models.baseline_cnn import BaselineCNN
from models.transfer_learning import get_transfer_model
from utils.gradcam import GradCAM
from api.database import init_db, get_or_create_patient, create_report, get_all_patients, get_patient_by_id, get_reports_by_patient_id

app = Flask(__name__)
# Enable CORS for all routes with exposed headers
CORS(app, expose_headers=['X-Prediction', 'X-Confidence', 'X-Mode', 'X-Prob-Benign', 'X-Prob-Malignant', 'X-Prob-Normal'])

# Initialize SQLite database
init_db()

# Create reports directory
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Create patients directory for storing all patient records
PATIENTS_DIR = os.path.join(REPORTS_DIR, 'patients')
os.makedirs(PATIENTS_DIR, exist_ok=True)

MODEL_TYPE = "resnet50"
# Get the absolute path to the models directory
# When running from backend directory: python api/app.py
# __file__ is api/app.py, so we need to go up one level to get backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "breast_cancer_detector_resnet50.pt")
NUM_CLASSES = 3
# IMPORTANT: Class names must match ImageFolder's alphabetical ordering used during training
# ImageFolder assigns labels alphabetically: Benign=0, Malignant=1, Normal=2
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug: Print the model path
print(f"Backend directory: {BACKEND_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

# Global model cache - load once at startup
_MODEL_CACHE = None
_TARGET_LAYER_CACHE = None


def load_model():
    """Load the model once and cache it for subsequent requests."""
    global _MODEL_CACHE, _TARGET_LAYER_CACHE
    
    # Return cached model if already loaded
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _TARGET_LAYER_CACHE
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        if MODEL_TYPE == "baseline":
            model = BaselineCNN(num_classes=NUM_CLASSES)
            target_layer = model.conv2
        else:
            model, target_layer = get_transfer_model(
                MODEL_TYPE, num_classes=NUM_CLASSES, pretrained=False
            )
        
        # Load the trained weights
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        # CRITICAL: Set to evaluation mode for BatchNorm layers
        model.eval()
        model.to(DEVICE)
        
        # Cache the model
        _MODEL_CACHE = model
        _TARGET_LAYER_CACHE = target_layer
        
        print(f"Model loaded successfully! Running on {DEVICE}")
        return model, target_layer
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Running in demo mode with random predictions.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def demo_prediction():
    """Generate a demo prediction for demonstration purposes"""
    pred_class = random.choice(CLASS_NAMES)
    conf_score = random.uniform(75.0, 95.0)
    return pred_class, conf_score


def predict_with_model(model, input_tensor):
    """Safe prediction with proper inference setup"""
    model.eval()  # Ensure evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, pred = torch.max(probabilities, 0)
        pred_class = CLASS_NAMES[pred.item()]
        conf_score = confidence.item() * 100
        # Get all class probabilities
        all_probs = {CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
    return pred_class, conf_score, pred.item(), all_probs


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API information and simple web interface"""
    # Check if user wants JSON response
    if request.headers.get("Accept") == "application/json":
        return jsonify(
            {
                "message": "Breast Cancer Detection API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "GET /": "API information (this endpoint)",
                    "GET /health": "Health check and model status",
                    "POST /predict": "Upload image for breast cancer detection",
                },
                "usage": {
                    "predict": "Send POST request to /predict with image file in form data",
                    "health": "Send GET request to /health for service status",
                    "web_interface": "Visit http://localhost:8502 for web interface",
                },
                "model_status": "demo" if load_model()[0] is None else "production",
                "device": str(DEVICE),
            }
        )

    # Return HTML interface
    html = (
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breast Cancer Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .form { background: #e8f5e8; padding: 20px; border-radius: 10px; }
            .status { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
            input[type="file"] { margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Breast Cancer Detection API</h1>
            <p>Upload a mammogram image to get a diagnosis prediction</p>
        </div>
        
        <div class="status">
            <strong>Status:</strong> Running<br>
            <strong>Model:</strong> """
        + ("Demo Mode" if load_model()[0] is None else "Production Mode")
        + """<br>
            <strong>Device:</strong> """
        + str(DEVICE)
        + """
        </div>
        
        <div class="form">
            <h3>📤 Upload Image for Prediction</h3>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br>
                <button type="submit">🔍 Get Prediction</button>
            </form>
        </div>
        
        <div class="endpoint">
            <h3>🔗 API Endpoints</h3>
            <p><strong>GET /</strong> - This page (API information)</p>
            <p><strong>GET /health</strong> - Health check and model status</p>
            <p><strong>POST /predict</strong> - Upload image for prediction</p>
        </div>
        
        <div class="endpoint">
            <h3>🌐 Web Interface</h3>
            <p>For a more advanced interface, visit: <a href="http://localhost:8502" target="_blank">http://localhost:8502</a></p>
        </div>
    </body>
    </html>
    """
    )
    return html


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    # Try to load model, fall back to demo mode
    model, target_layer = load_model()

    if model is not None:
        # Real prediction with proper inference setup
        input_tensor = preprocess_image(image).to(DEVICE)
        pred_class, conf_score, pred_idx, all_probs = predict_with_model(model, input_tensor)

        # Real Grad-CAM with proper inference setup
        try:
            gradcam = GradCAM(model, target_layer)
            model.eval()  # Ensure evaluation mode for GradCAM
            with torch.no_grad():
                cam = gradcam(input_tensor, class_idx=pred_idx)
            heatmap = np.uint8(255 * cam)
            heatmap_img = Image.fromarray(heatmap).resize((224, 224)).convert("RGB")
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            heatmap_img = image.resize((224, 224))
    else:
        # Demo prediction
        pred_class, conf_score = demo_prediction()
        all_probs = {name: round(random.uniform(10, 40), 2) for name in CLASS_NAMES}
        all_probs[pred_class] = round(conf_score, 2)
        # Demo heatmap (just resize the original image)
        heatmap_img = image.resize((224, 224))

    # Save heatmap to buffer
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    buf.seek(0)

    return (
        send_file(
            buf, mimetype="image/png", as_attachment=False, download_name="heatmap.png"
        ),
        200,
        {
            "X-Prediction": pred_class,
            "X-Confidence": f"{conf_score:.2f}",
            "X-Mode": "demo" if model is None else "production",
            "X-Prob-Benign": str(all_probs.get("Benign", 0)),
            "X-Prob-Malignant": str(all_probs.get("Malignant", 0)),
            "X-Prob-Normal": str(all_probs.get("Normal", 0)),
        },
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": load_model()[0] is not None,
            "device": str(DEVICE),
        }
    )


@app.route("/generate-report", methods=["POST"])
def generate_report():
    """Generate and save a report with patient information"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract patient info and results
        patient_info = data.get("patientInfo", {})
        prediction = data.get("prediction", "Unknown")
        confidence = data.get("confidence", 0)
        probabilities = data.get("probabilities", {})
        analysis_date = data.get("analysisDate", datetime.datetime.now().isoformat())
        
        # Generate unique report ID
        patient_name = patient_info.get('lastName', 'Patient').replace(' ', '_')
        report_id = f"BC_{patient_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save patient to SQLite database
        patient_id = get_or_create_patient(patient_info)
        
        # Generate PDF report
        pdf_filename = f"{report_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        # Create PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=1)
        story.append(Paragraph("Breast Cancer Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report Info
        story.append(Paragraph("Report Information", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        report_info = [
            ["Report ID:", report_id],
            ["Analysis Date:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        t = Table(report_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("Patient Information", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        patient_data = [
            ["Name:", f"{patient_info.get('firstName', '')} {patient_info.get('lastName', '')}"],
            ["Date of Birth:", patient_info.get('dateOfBirth', 'N/A')],
            ["Gender:", patient_info.get('gender', 'Female')],
            ["Phone:", patient_info.get('phone', 'N/A')],
            ["Email:", patient_info.get('email', 'N/A')],
            ["Reason for Exam:", patient_info.get('reasonForExam', 'N/A')],
            ["Previous Mammogram:", patient_info.get('previousMammogram', 'N/A')],
            ["Family History:", patient_info.get('familyHistory', 'N/A')],
        ]
        t = Table(patient_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Analysis Results
        story.append(Paragraph("Analysis Results", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Determine color based on prediction
        if prediction == 'Normal':
            result_color = colors.green
        elif prediction == 'Benign':
            result_color = colors.orange
        else:
            result_color = colors.red
        
        result_data = [
            ["Prediction:", prediction],
            ["Confidence:", f"{confidence}%"],
        ]
        t = Table(result_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))
        
        # Probability Distribution
        story.append(Paragraph("Probability Distribution", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        prob_data = [
            ["Class", "Probability"],
            ["Normal", f"{probabilities.get('Normal', 0)}%"],
            ["Benign", f"{probabilities.get('Benign', 0)}%"],
            ["Malignant", f"{probabilities.get('Malignant', 0)}%"],
        ]
        t = Table(prob_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 30))
        
        # Disclaimer
        story.append(Paragraph("Disclaimer", styles['Heading3']))
        disclaimer_text = """This report is generated by an AI system for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified healthcare provider for medical decisions."""
        story.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Save report to SQLite database
        create_report(
            patient_id=patient_id,
            report_id=report_id,
            prediction=prediction,
            confidence=float(confidence),
            probabilities=probabilities,
            pdf_path=pdf_filename
        )
        
        return jsonify({
            "success": True,
            "report_id": report_id,
            "pdf_url": f"/reports/{pdf_filename}",
            "message": "Report saved successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports/<filename>")
def serve_report(filename):
    """Serve PDF report files"""
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/patients", methods=["GET"])
def get_patients():
    """Get all patients with their analysis history from SQLite database"""
    try:
        patients = get_all_patients()
        
        # Format patients for frontend
        formatted_patients = []
        for patient in patients:
            formatted_patient = {
                'id': patient['id'],
                'firstName': patient['first_name'],
                'lastName': patient['last_name'],
                'dateOfBirth': patient['date_of_birth'],
                'gender': patient['gender'],
                'phone': patient['phone'],
                'email': patient['email'],
                'address': patient['address'],
                'reasonForExam': patient['reason_for_exam'],
                'previousMammogram': patient['previous_mammogram'],
                'familyHistory': patient['family_history'],
                'reports': []
            }
            
            # Format reports
            for report in patient.get('reports', []):
                formatted_patient['reports'].append({
                    'report_id': report['report_id'],
                    'analysis_date': report['analysis_date'],
                    'prediction': report['prediction'],
                    'confidence': report['confidence'],
                    'probabilities': {
                        'Normal': report['probabilities_normal'],
                        'Benign': report['probabilities_benign'],
                        'Malignant': report['probabilities_malignant']
                    },
                    'pdf_path': report['pdf_path']
                })
            
            formatted_patients.append(formatted_patient)
        
        return jsonify({
            "success": True,
            "patients": formatted_patients,
            "count": len(formatted_patients)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports", methods=["GET"])
def get_reports():
    """Get all saved reports"""
    try:
        reports = []
        for filename in os.listdir(REPORTS_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(REPORTS_DIR, filename), 'r') as f:
                    reports.append(json.load(f))
        
        # Sort by date, newest first
        reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "reports": reports,
            "count": len(reports)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports/<report_id>", methods=["GET"])
def get_report(report_id):
    """Get a specific report by ID"""
    try:
        report_file = os.path.join(REPORTS_DIR, f"{report_id}.json")
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Report not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Breast Cancer Detection API...")
    print("API will run in demo mode if no model is found")
    print("API will be available at http://localhost:5000")
    
    # Pre-load the model at startup
    print("Pre-loading model...")
    model, target_layer = load_model()
    if model is not None:
        print("Model pre-loaded successfully!")
    else:
        print("Warning: Model could not be loaded. Running in demo mode.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
