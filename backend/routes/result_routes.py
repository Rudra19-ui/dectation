import os
import json
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from models.inference import load_model, predict_with_uncertainty
from utils.gradcam import GradCAM
from utils.visualize import create_visualization
from utils.report import DiagnosisReport

result_bp = Blueprint('result', __name__)

@result_bp.route('/result/<image_id>', methods=['GET'])
def show_result(image_id):
    """
    Display the analysis result page for a given image.
    """
    # In a real application, we would retrieve the image and results from a database
    # For this example, we'll use session data or mock data
    
    result_data = session.get('result_data')
    
    if not result_data:
        # Mock data for demonstration
        result_data = {
            "prediction": "Malignant",
            "confidence": 82,
            "probabilities": {"Normal": 5, "Benign": 13, "Malignant": 82},
            "model_version": "v1.2",
            "explain_id": f"exp_{image_id}",
            "report_id": f"rep_{image_id}",
            "original_image_url": f"/uploads/{image_id}.png",
            "heatmap_url": f"/reports/heatmap_overlay_{image_id}.png"
        }
    
    return render_template('result_page.html', result=result_data)

@result_bp.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    API endpoint to analyze an uploaded image and return results.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the uploaded image
    image_id = os.urandom(4).hex()
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    image_path = os.path.join(upload_dir, f"{image_id}.png")
    image_file.save(image_path)
    
    # Load model and make prediction
    model = load_model()
    if model is None:
        return jsonify({'error': 'Failed to load model'}), 500
    
    result = predict_with_uncertainty(model, image_path)
    if result is None:
        return jsonify({'error': 'Failed to analyze image'}), 500
    
    # Generate GradCAM visualization
    try:
        # Get the last convolutional layer for GradCAM
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            raise ValueError("Could not find convolutional layer for GradCAM")
        
        # Apply GradCAM
        from models.inference import preprocess_image
        import torch
        input_tensor = preprocess_image(image_path)
        gradcam = GradCAM(model, target_layer)
        class_idx = list(result["probabilities"].keys()).index(result["prediction"])
        cam = gradcam(input_tensor, class_idx=class_idx)
        
        # Create visualization
        output_dir = os.path.join(os.getcwd(), 'reports')
        vis_result = create_visualization(image_path, cam, output_dir=output_dir)
        
        # Add visualization info to result
        result["explain_id"] = vis_result["explain_id"]
        result["explain_url"] = vis_result["explain_url"]
        result["heatmap_url"] = vis_result["heatmap_overlay_path"]
    except Exception as e:
        print(f"Error in GradCAM visualization: {e}")
        # Continue without visualization if it fails
        result["explain_id"] = f"exp_{image_id}"
        result["explain_url"] = ""
        result["heatmap_url"] = ""
    
    # Generate PDF report
    try:
        # Create patient info (in a real app, this would come from the user/database)
        from datetime import datetime
        patient_info = {
            "patient_id": f"ANON{image_id}",
            "age": 45,  # Example age
            "gender": "F",  # Example gender
            "scan_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Generate report
        report = DiagnosisReport()
        report_result = report.generate(
            original_image_path=image_path,
            heatmap_path=result.get("heatmap_url", ""),
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            patient_info=patient_info,
            model_version=result.get("model_version", "v1.0")
        )
        
        # Add report info to result
        result["report_id"] = report_result["report_id"]
        result["report_url"] = report_result["report_url"]
    except Exception as e:
        print(f"Error in report generation: {e}")
        # Continue without report if it fails
        result["report_id"] = f"rep_{image_id}"
        result["report_url"] = ""
    
    # Add recommendation if confidence is low
    if result["confidence"] < 0.5:
        result["recommendation"] = "Review by radiologist"
    
    # Store result in session for the result page
    session['result_data'] = result
    
    # Return JSON result and redirect URL
    return jsonify({
        'result': result,
        'redirect': url_for('result.show_result', image_id=image_id)
    })

@result_bp.route('/api/share-report', methods=['POST'])
def share_report():
    """
    API endpoint to share a report with a radiologist.
    """
    data = request.json
    if not data or 'report_id' not in data:
        return jsonify({'error': 'No report ID provided'}), 400
    
    # In a real application, this would send the report via email or webhook
    # and log the action in the audit trail
    
    # Mock audit log entry
    from datetime import datetime
    audit_entry = {
        'user_id': data.get('user_id', 'anonymous'),
        'action': 'share_report',
        'request_id': data.get('request_id', os.urandom(4).hex()),
        'timestamp': datetime.now().isoformat(),
        'model_version': data.get('model_version', 'v1.0'),
        'resource_id': data['report_id']
    }
    
    # In a real application, save this to a database
    print(f"Audit log entry: {json.dumps(audit_entry)}")
    
    return jsonify({
        'success': True,
        'message': 'Report shared successfully',
        'audit_id': audit_entry['request_id']
    })

@result_bp.route('/api/delete-data', methods=['POST'])
def delete_data():
    """
    API endpoint to delete user data.
    """
    data = request.json
    if not data or 'image_id' not in data:
        return jsonify({'error': 'No image ID provided'}), 400
    
    # In a real application, this would delete the image, results, and reports
    # and log the action in the audit trail
    
    # Mock deletion and audit log
    from datetime import datetime
    audit_entry = {
        'user_id': data.get('user_id', 'anonymous'),
        'action': 'delete_data',
        'request_id': data.get('request_id', os.urandom(4).hex()),
        'timestamp': datetime.now().isoformat(),
        'resource_id': data['image_id']
    }
    
    # In a real application, save this to a database
    print(f"Audit log entry: {json.dumps(audit_entry)}")
    
    # Clear session data
    if 'result_data' in session:
        session.pop('result_data')
    
    return jsonify({
        'success': True,
        'message': 'Data deleted successfully',
        'audit_id': audit_entry['request_id']
    })

@result_bp.route('/api/flag-incorrect', methods=['POST'])
def flag_incorrect():
    """
    API endpoint to flag a result as incorrect for active learning.
    """
    data = request.json
    if not data or 'image_id' not in data:
        return jsonify({'error': 'No image ID provided'}), 400
    
    # In a real application, this would flag the result for review
    # and log the action in the audit trail
    
    # Mock flagging and audit log
    from datetime import datetime
    audit_entry = {
        'user_id': data.get('user_id', 'anonymous'),
        'action': 'flag_incorrect',
        'request_id': data.get('request_id', os.urandom(4).hex()),
        'timestamp': datetime.now().isoformat(),
        'model_version': data.get('model_version', 'v1.0'),
        'resource_id': data['image_id'],
        'feedback': data.get('feedback', '')
    }
    
    # In a real application, save this to a database
    print(f"Audit log entry: {json.dumps(audit_entry)}")
    
    return jsonify({
        'success': True,
        'message': 'Result flagged for review',
        'audit_id': audit_entry['request_id']
    })