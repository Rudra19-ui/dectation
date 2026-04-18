#!/usr/bin/env python3
"""
Breast Cancer Classification - Backend API
FastAPI server with REST endpoints for image prediction
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms
from utils.async_jobs import JobStatus, enqueue_large_image_job, job_queue
from utils.explainability import process_image_with_gradcam
from model_files.predictor import BreastCancerPredictor
from model_files.thermal_predictor import ThermalImagePredictor

from utils.validation import (
    get_file_metadata,
    validate_upload_bytes,
    validation_error_to_http_exception,
)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="AI-powered breast cancer classification from mammogram images",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize predictor - now imported from model.predictor


# Initialize predictors
predictor = BreastCancerPredictor()
thermal_predictor = ThermalImagePredictor()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "mammography": {
                "loaded": predictor.model is not None,
                "type": "ResNet50",
                "classes": predictor.class_names
            },
            "thermal": {
                "loaded": thermal_predictor.model is not None,
                "type": "ResNet50",
                "classes": thermal_predictor.class_names
            }
        },
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /model-info": "Mammography model information",
            "GET /thermal-model-info": "Thermal model information",
            "POST /predict": "Predict mammography image",
            "POST /predict-thermal": "Predict thermal image",
            "POST /predict-batch": "Predict multiple images",
            "GET /job/{job_id}": "Get status of async job",
            "POST /explain": "Generate Grad-CAM visualization for an image"
        },
        "usage": {
            "mammography_prediction": "POST /predict with mammography image file",
            "thermal_prediction": "POST /predict-thermal with thermal image file",
            "batch_prediction": "POST /predict-batch with multiple image files",
            "async_jobs": "Large images are processed asynchronously, check job status with GET /job/{job_id}",
            "explainability": "POST /explain with image file to get Grad-CAM visualization"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor.model is not None,
        "device": str(predictor.device),
    }


@app.post("/predict")
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    consent: bool = Form(True),
    request_id: Optional[str] = Form(None),
):
    """Predict breast cancer from uploaded image"""
    try:
        # Verify consent
        if not consent:
            raise HTTPException(status_code=400, detail="Consent is required for image processing")
        
        # Generate request_id if not provided
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Log consent
        from utils.consent_log import consent_logger
        # Get client IP (in a real implementation, you would get this from the request)
        client_host = "127.0.0.1"  # Default for local testing
        consent_logger.log_consent(request_id, consent, client_host)
        
        # Read file bytes
        file_bytes = await file.read()

        # Validate upload
        try:
            validation_result = validate_upload_bytes(file_bytes)
        except Exception as e:
            raise validation_error_to_http_exception(e)

        # Extract image from validation result
        if validation_result["is_dicom"]:
            # Convert DICOM to PIL Image
            ds = validation_result["dicom_dataset"]
            pixel_array = ds.pixel_array

            # Normalize pixel values
            if pixel_array.dtype != np.uint8:
                pixel_array = (
                    (pixel_array - pixel_array.min())
                    / (pixel_array.max() - pixel_array.min())
                    * 255
                ).astype(np.uint8)

            # Convert to PIL Image
            if len(pixel_array.shape) == 3:
                image = Image.fromarray(pixel_array)
            else:
                image = Image.fromarray(pixel_array, mode="L").convert("RGB")
        else:
            # Use validated image
            image = validation_result["image"]

        # Convert image to bytes for prediction
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()
        
        # Check if image is large (> 5MB) and should be processed asynchronously
        if len(image_bytes) > 5 * 1024 * 1024 and background_tasks is not None:
            # Process large image asynchronously
            metadata = get_file_metadata(file_bytes, file.filename)
            job_id = enqueue_large_image_job(
                image_bytes,
                lambda img: {
                    **predictor.predict_image(img),
                    "filename": metadata["filename"],
                    "file_size": metadata["file_size"],
                    "content_type": validation_result["mime_type"],
                    "file_type": "DICOM" if validation_result["is_dicom"] else "Image",
                    "validation_passed": True,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                    "consent_given": consent
                },
                background_tasks
            )
            return {"job_id": job_id, "status": "processing", "request_id": request_id}
        else:
            # Make prediction
            result = predictor.predict_image(image_bytes)

            # Add metadata
            metadata = get_file_metadata(file_bytes, file.filename)
            result.update(
                {
                    "filename": metadata["filename"],
                    "file_size": metadata["file_size"],
                    "content_type": validation_result["mime_type"],
                    "file_type": "DICOM" if validation_result["is_dicom"] else "Image",
                    "validation_passed": True,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id,
                    "consent_given": consent
                }
            )

            # Add DICOM-specific metadata if applicable
            if validation_result["is_dicom"]:
                result.update(
                    {
                        "study_uid": validation_result.get("study_uid"),
                        "series_uid": validation_result.get("series_uid"),
                        "instance_uid": validation_result.get("instance_uid"),
                        "has_phi": validation_result.get("has_phi", False),
                    }
                )

            return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    return {
        "model_type": "ResNet50",
        "classes": predictor.class_names,
        "device": str(predictor.device),
        "model_path": predictor.model_path,
        "model_loaded": predictor.model is not None,
        "explainability": {
            "grad_cam_available": True,
            "roi_filtering_available": True,
        }
    }


@app.get("/thermal-model-info")
async def get_thermal_model_info():
    """Get thermal model information"""
    return thermal_predictor.get_model_info()


@app.post("/predict-thermal")
async def predict_thermal_image(
    file: UploadFile = File(...),
    consent: bool = Form(True),
    request_id: Optional[str] = Form(None),
):
    """Predict breast cancer from thermal image"""
    try:
        # Verify consent
        if not consent:
            raise HTTPException(status_code=400, detail="Consent is required for image processing")
        
        # Generate request_id if not provided
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Log consent
        from utils.consent_log import consent_logger
        client_host = "127.0.0.1"  # Default for local testing
        consent_logger.log_consent(request_id, consent, client_host)
        
        # Read file bytes
        file_bytes = await file.read()

        # Validate upload
        try:
            validation_result = validate_upload_bytes(file_bytes)
        except Exception as e:
            raise validation_error_to_http_exception(e)

        # Extract PIL image (same as /predict — supports DICOM and raster)
        if validation_result["is_dicom"]:
            ds = validation_result["dicom_dataset"]
            pixel_array = ds.pixel_array
            if pixel_array.dtype != np.uint8:
                pixel_array = (
                    (pixel_array - pixel_array.min())
                    / (pixel_array.max() - pixel_array.min())
                    * 255
                ).astype(np.uint8)
            if len(pixel_array.shape) == 3:
                image = Image.fromarray(pixel_array)
            else:
                image = Image.fromarray(pixel_array, mode="L").convert("RGB")
        else:
            image = validation_result["image"]
        
        # Convert image to bytes for prediction
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        # Make prediction
        result = thermal_predictor.predict_image(image_bytes)

        # Add metadata
        metadata = get_file_metadata(file_bytes, file.filename)
        result.update(
            {
                "filename": metadata["filename"],
                "file_size": metadata["file_size"],
                "content_type": validation_result["mime_type"],
                "file_type": "Thermal Image",
                "validation_passed": True,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "consent_given": consent,
                "scan_type": "thermal"
            }
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thermal prediction error: {str(e)}")


@app.post("/predict-async")
async def predict_image_async(
    file: UploadFile = File(...),
    consent: bool = Form(True),
    request_id: Optional[str] = Form(None),
):
    """Endpoint for asynchronous image prediction.
    
    This endpoint is optimized for larger files or batch processing. It immediately returns a job ID
    that can be used to poll for results.
    """
    try:
        # Verify consent
        if not consent:
            raise HTTPException(status_code=400, detail="Consent is required for image processing")
        
        # Generate request_id if not provided
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Log consent
        from utils.consent_log import consent_logger
        client_host = "127.0.0.1"  # Default for local testing
        consent_logger.log_consent(request_id, consent, client_host)
        
        # Read file bytes
        file_bytes = await file.read()

        # Validate upload
        try:
            validation_result = validate_upload_bytes(file_bytes)
        except Exception as e:
            raise validation_error_to_http_exception(e)
            
        # Use request_id as job_id for consistency
        job_id = request_id
        
        # Enqueue the job
        background_tasks = BackgroundTasks()
        job_id = enqueue_large_image_job(
            file_bytes,
            lambda img: {
                **predictor.predict_image(img),
                "filename": file.filename,
                "file_size": len(file_bytes),
                "content_type": validation_result["mime_type"],
                "file_type": "DICOM" if validation_result["is_dicom"] else "Image",
                "validation_passed": True,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "consent_given": consent
            },
            background_tasks
        )
        
        return {"status": "processing", "job_id": job_id, "request_id": request_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """Predict multiple images at once"""
    try:
        results = []
        jobs = []

        for file in files:
            if not file.content_type.startswith("image/"):
                continue

            image_bytes = await file.read()
            
            # Check if image is large (> 5MB) and should be processed asynchronously
            if len(image_bytes) > 5 * 1024 * 1024 and background_tasks is not None:
                # Process large image asynchronously
                job_id = enqueue_large_image_job(
                    image_bytes,
                    lambda img: {
                        **predictor.predict_image(img),
                        "filename": file.filename,
                        "file_size": len(image_bytes)
                    },
                    background_tasks
                )
                jobs.append({"filename": file.filename, "job_id": job_id})
            else:
                result = predictor.predict_image(image_bytes)
                result.update({"filename": file.filename, "file_size": len(image_bytes)})
                results.append(result)

        return {
            "predictions": results,
            "total_images": len(results),
            "jobs": jobs,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api-docs")
async def get_api_docs():
    """Get API documentation"""
    return {
        "endpoints": {
            "GET /": "Root endpoint with API info",
            "GET /health": "Health check",
            "GET /model-info": "Model information",
            "POST /predict": "Predict single image",
            "POST /predict-batch": "Predict multiple images",
            "GET /job/{job_id}": "Get status of async job",
            "POST /explain": "Generate Grad-CAM visualization for an image"
        },
        "usage": {
            "single_prediction": "POST /predict with image file",
            "batch_prediction": "POST /predict-batch with multiple image files",
            "async_jobs": "Large images are processed asynchronously, check job status with GET /job/{job_id}",
            "explainability": "POST /explain with image file to get Grad-CAM visualization"
        },
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an asynchronous job"""
    job = job_queue.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status == JobStatus.COMPLETED:
        return {"status": "completed", "result": job.result}
    elif job.status == JobStatus.FAILED:
        return {"status": "failed", "error": job.error}
    else:
        return {"status": "processing"}

@app.post("/explain")
async def explain_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Generate Grad-CAM visualization for an image"""
    try:
        # Read file bytes
        file_bytes = await file.read()

        # Validate upload
        try:
            validation_result = validate_upload_bytes(file_bytes, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Extract image from validation result
        if validation_result["is_dicom"]:
            # Convert DICOM to PIL Image
            ds = validation_result["dicom_dataset"]
            pixel_array = ds.pixel_array

            # Normalize pixel values
            if pixel_array.dtype != np.uint8:
                pixel_array = (
                    (pixel_array - pixel_array.min())
                    / (pixel_array.max() - pixel_array.min())
                    * 255
                ).astype(np.uint8)

            # Convert to PIL Image
            if len(pixel_array.shape) == 3:
                image = Image.fromarray(pixel_array)
            else:
                image = Image.fromarray(pixel_array, mode="L").convert("RGB")
            
            # Convert image to bytes for processing
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            image_data = img_buffer.getvalue()
        else:
            # Use validated image
            image = validation_result["image"]
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            image_data = img_buffer.getvalue()

        # Get the original image as numpy array for visualization
        image_pil = Image.open(io.BytesIO(image_data))
        original_image = np.array(image_pil)
        
        # Prepare image tensor for model
        image_tensor = predictor.preprocess_image(image_data)
        
        # Get the target layer for Grad-CAM
        # For most CNN models, the last convolutional layer is a good choice
        # This is model-specific and might need adjustment
        if hasattr(predictor.model, 'features'):
            # For models like VGG, AlexNet
            target_layer = predictor.model.features[-1]
        elif hasattr(predictor.model, 'layer4'):
            # For ResNet models
            target_layer = predictor.model.layer4[-1]
        else:
            # Fallback
            modules = list(predictor.model.modules())
            target_layer = [m for m in modules if isinstance(m, nn.Conv2d)][-1]
        
        # Process with Grad-CAM
        result = process_image_with_gradcam(
            model=predictor.model,
            target_layer=target_layer,
            image_tensor=image_tensor,
            original_image=original_image
        )
        
        # Add prediction result
        prediction = predictor.predict_image(image_data)
        result.update(prediction)
        
        # Add metadata
        metadata = get_file_metadata(file_bytes, file.filename)
        result.update({
            "filename": metadata["filename"],
            "file_size": metadata["file_size"],
            "content_type": validation_result["mime_type"],
            "file_type": "DICOM" if validation_result["is_dicom"] else "Image",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Add DICOM-specific metadata if applicable
        if validation_result["is_dicom"]:
            result.update({
                "study_uid": validation_result.get("study_uid"),
                "series_uid": validation_result.get("series_uid"),
                "instance_uid": validation_result.get("instance_uid"),
                "has_phi": validation_result.get("has_phi", False),
            })
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}\n{traceback_str}")


# ============== Report Generation Endpoints ==============

# Import database and PDF generation
import sys
api_dir = Path(__file__).parent / "api"
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

# Reports directory
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Import database functions
try:
    from api.database import init_db, get_or_create_patient, create_report, get_all_patients
except ImportError:
    # Fallback if api module not found
    init_db = None
    get_or_create_patient = None
    create_report = None
    get_all_patients = None

# PDF Generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will not be available.")


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    if init_db:
        try:
            init_db()
            print("Database initialized successfully")
        except Exception as e:
            print(f"Database initialization error: {e}")


@app.post("/generate-report")
async def generate_report(request: dict):
    """Generate and save a PDF report with patient information"""
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF generation not available. Install reportlab.")
    
    if not get_or_create_patient:
        raise HTTPException(status_code=500, detail="Database functions not available")
    
    try:
        data = request
        
        # Extract patient info and results
        patient_info = data.get("patientInfo", {})
        prediction = data.get("prediction", "Unknown")
        confidence = data.get("confidence", 0)
        probabilities = data.get("probabilities", {})
        scan_type = data.get("scanType", "mammography")
        analysis_date = data.get("analysisDate", datetime.now().isoformat())
        
        # Generate unique report ID
        patient_name = patient_info.get('lastName', 'Patient').replace(' ', '_')
        report_id = f"BC_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save patient to database
        patient_id = get_or_create_patient(patient_info)
        
        # Generate PDF report
        pdf_filename = f"{report_id}.pdf"
        pdf_path = REPORTS_DIR / pdf_filename
        
        # Create PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
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
            ["Analysis Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Scan Type:", scan_type.title()],
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
        
        if scan_type.lower() == "thermal":
            prob_data = [
                ["Class", "Probability"],
                ["Healthy", f"{probabilities.get('Healthy', probabilities.get('healthy', 0))}%"],
                ["Sick", f"{probabilities.get('Sick', probabilities.get('sick', 0))}%"],
            ]
        else:
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
        
        # Save report to database
        create_report(
            patient_id=patient_id,
            report_id=report_id,
            prediction=prediction,
            confidence=float(confidence),
            probabilities=probabilities,
            pdf_path=pdf_filename
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "pdf_url": f"/reports/{pdf_filename}",
            "message": "Report saved successfully"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/reports/{filename}")
async def serve_report(filename: str):
    """Serve PDF report files"""
    from fastapi.responses import FileResponse
    pdf_path = REPORTS_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


@app.get("/reports")
async def list_reports():
    """List all reports"""
    if not get_all_patients:
        raise HTTPException(status_code=500, detail="Database functions not available")
    
    try:
        patients = get_all_patients()
        reports = []
        for patient in patients:
            for report in patient.get('reports', []):
                reports.append({
                    'report_id': report['report_id'],
                    'patient_name': f"{patient['first_name']} {patient['last_name']}",
                    'prediction': report['prediction'],
                    'confidence': report['confidence'],
                    'analysis_date': report['analysis_date'],
                    'pdf_path': report['pdf_path']
                })
        return {"reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients")
async def list_patients():
    """Get all patients with their analysis history"""
    if not get_all_patients:
        raise HTTPException(status_code=500, detail="Database functions not available")
    
    try:
        patients = get_all_patients()
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
        
        return {"patients": formatted_patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting Breast Cancer Classification API Server...")
    print(f"Using device: {predictor.device}")
    print(f"Model loaded: {predictor.model is not None}")

    try:
        uvicorn.run(
            "backend_api:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        print(f"Server startup error: {e}")
        print(
            "Try running: python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000"
        )
