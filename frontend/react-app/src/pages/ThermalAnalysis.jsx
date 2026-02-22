import { useState, useRef } from 'react';
import { 
  Upload, Image, FileText, AlertCircle, CheckCircle, 
  Loader, Download, RotateCcw, Info, User, Calendar, 
  Phone, Mail, File, History, Thermometer
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { API_URL } from '../config/api';
import './Analysis.css';

const ThermalAnalysis = () => {
  const [step, setStep] = useState(1); // 1: Patient Info, 2: Image Upload, 3: Results
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [formErrors, setFormErrors] = useState({});
  const fileInputRef = useRef(null);

  const [patientInfo, setPatientInfo] = useState({
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    gender: 'Female',
    phone: '',
    email: '',
    address: '',
    reasonForExam: '',
    previousThermogram: 'No',
    familyHistory: 'No'
  });

  // Validation functions
  const validateName = (name) => {
    // Only allow letters and spaces
    const nameRegex = /^[A-Za-z\s]+$/;
    return nameRegex.test(name);
  };

  const validatePhone = (phone) => {
    // Exactly 10 digits
    const phoneRegex = /^\d{10}$/;
    return phoneRegex.test(phone);
  };

  const validateEmail = (email) => {
    if (!email) return true; // Email is optional
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handlePatientInfoChange = (e) => {
    const { name, value } = e.target;
    setPatientInfo(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const handleFileSelect = (file) => {
    if (file) {
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
      if (!validTypes.includes(file.type)) {
        setError('Please upload a valid image file (PNG or JPEG)');
        return;
      }

      // Check if this looks like a mammography image (DICOM or typical mammography naming)
      const fileName = file.name.toLowerCase();
      if (file.name.endsWith('.dcm') || 
          fileName.includes('mammogram') || 
          fileName.includes('mammo') ||
          fileName.includes('dicom')) {
        setError('This appears to be a mammography image. Please upload a thermal image instead, or use the Mammography Analysis page.');
        return;
      }

      setSelectedFile(file);
      setError(null);

      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => setPreviewUrl(e.target.result);
        reader.readAsDataURL(file);
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleUploadClick = (e) => {
    // Prevent event from bubbling up
    e.stopPropagation();
    // Reset the input value to allow selecting the same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const proceedToUpload = () => {
    const errors = {};
    
    // Validate first name
    if (!patientInfo.firstName) {
      errors.firstName = 'First name is required';
    } else if (!validateName(patientInfo.firstName)) {
      errors.firstName = 'First name should only contain letters';
    }
    
    // Validate last name
    if (!patientInfo.lastName) {
      errors.lastName = 'Last name is required';
    } else if (!validateName(patientInfo.lastName)) {
      errors.lastName = 'Last name should only contain letters';
    }
    
    // Validate date of birth
    if (!patientInfo.dateOfBirth) {
      errors.dateOfBirth = 'Date of birth is required';
    }
    
    // Validate phone (required)
    if (!patientInfo.phone) {
      errors.phone = 'Phone number is required';
    } else if (!validatePhone(patientInfo.phone)) {
      errors.phone = 'Phone number must be exactly 10 digits';
    }
    
    // Validate email (required)
    if (!patientInfo.email) {
      errors.email = 'Email address is required';
    } else if (!validateEmail(patientInfo.email)) {
      errors.email = 'Please enter a valid email address';
    }
    
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors);
      setError('Please correct the errors in the form');
      return;
    }
    
    setFormErrors({});
    setError(null);
    setStep(2);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_URL}/predict-thermal`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }
      
      const data = await response.json();
      console.log('Thermal prediction from backend:', data);
      
      // Convert probabilities to percentages (backend returns 0-1 range)
      const healthyProb = data.probabilities ? Math.round(data.probabilities[0] * 100) : 0;
      const sickProb = data.probabilities ? Math.round(data.probabilities[1] * 100) : 0;
      
      // Confidence is already in 0-1 range, convert to percentage
      const confidencePercent = data.confidence ? Math.round(data.confidence * 100) : 75;
      
      const analysisResults = {
        prediction: data.prediction.charAt(0).toUpperCase() + data.prediction.slice(1), // Capitalize
        confidence: confidencePercent,
        probabilities: {
          Healthy: healthyProb,
          Sick: sickProb
        },
        patientInfo: patientInfo,
        analysisDate: new Date().toISOString(),
        scanType: 'thermal'
      };
      setResult(analysisResults);
      setStep(3);
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = async () => {
    try {
      console.log('Sending thermal report data:', result);
      
      const response = await fetch(`${API_URL}/generate-report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({...result, scanType: 'thermal'}),
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      if (data.success && data.pdf_url) {
        console.log('Fetching PDF from:', `${API_URL}${data.pdf_url}`);
        const pdfResponse = await fetch(`${API_URL}${data.pdf_url}`);
        
        if (!pdfResponse.ok) {
          throw new Error(`PDF fetch error! status: ${pdfResponse.status}`);
        }
        
        const blob = await pdfResponse.blob();
        console.log('Blob type:', blob.type, 'Size:', blob.size);
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `Thermal_Report_${patientInfo.lastName}_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(link);
        console.log('PDF download complete');
      } else {
        throw new Error('PDF generation failed: ' + JSON.stringify(data));
      }
    } catch (err) {
      console.error('PDF generation failed:', err);
      alert('PDF generation failed. Please check console for details.');
    }
  };

  const resetAnalysis = () => {
    setStep(1);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setPatientInfo({
      firstName: '',
      lastName: '',
      dateOfBirth: '',
      gender: 'Female',
      phone: '',
      email: '',
      address: '',
      reasonForExam: '',
      previousThermogram: 'No',
      familyHistory: 'No'
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getResultClass = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'healthy': return 'result-normal';
      case 'sick': return 'result-malignant';
      default: return '';
    }
  };

  const getResultIcon = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'healthy': return <CheckCircle className="result-icon success" />;
      case 'sick': return <AlertCircle className="result-icon danger" />;
      default: return null;
    }
  };

  return (
    <div className="analysis-page">
      <div className="container">
        {/* Header Banner */}
        <div className="thermal-banner">
          <Thermometer size={32} />
          <div>
            <h2>Thermal Imaging Analysis</h2>
            <p>AI-powered breast cancer screening using thermal imaging</p>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="progress-steps">
          <div className={`step ${step >= 1 ? 'active' : ''} ${step > 1 ? 'completed' : ''}`}>
            <div className="step-number">1</div>
            <div className="step-label">Patient Info</div>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 2 ? 'active' : ''} ${step > 2 ? 'completed' : ''}`}>
            <div className="step-number">2</div>
            <div className="step-label">Upload Image</div>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 3 ? 'active' : ''}`}>
            <div className="step-number">3</div>
            <div className="step-label">Results</div>
          </div>
        </div>

        {/* Step 1: Patient Information */}
        {step === 1 && (
          <div className="patient-info-section">
            <div className="section-header">
              <h1>Patient Information</h1>
              <p>Please enter patient details before proceeding with the thermal imaging analysis</p>
            </div>

            <div className="patient-form">
              <div className="form-section">
                <h3><User /> Personal Information</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>First Name *</label>
                    <input
                      type="text"
                      name="firstName"
                      value={patientInfo.firstName}
                      onChange={handlePatientInfoChange}
                      placeholder="Enter first name"
                      className={formErrors.firstName ? 'error' : ''}
                    />
                    {formErrors.firstName && <span className="field-error">{formErrors.firstName}</span>}
                  </div>
                  <div className="form-group">
                    <label>Last Name *</label>
                    <input
                      type="text"
                      name="lastName"
                      value={patientInfo.lastName}
                      onChange={handlePatientInfoChange}
                      placeholder="Enter last name"
                      className={formErrors.lastName ? 'error' : ''}
                    />
                    {formErrors.lastName && <span className="field-error">{formErrors.lastName}</span>}
                  </div>
                  <div className="form-group">
                    <label>Date of Birth *</label>
                    <input
                      type="date"
                      name="dateOfBirth"
                      value={patientInfo.dateOfBirth}
                      onChange={handlePatientInfoChange}
                      className={formErrors.dateOfBirth ? 'error' : ''}
                    />
                    {formErrors.dateOfBirth && <span className="field-error">{formErrors.dateOfBirth}</span>}
                  </div>
                  <div className="form-group">
                    <label>Gender</label>
                    <select
                      name="gender"
                      value="Female"
                      disabled
                    >
                      <option value="Female">Female</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="form-section">
                <h3><Phone /> Contact Information</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Phone Number *</label>
                    <input
                      type="tel"
                      name="phone"
                      value={patientInfo.phone}
                      onChange={handlePatientInfoChange}
                      placeholder="Enter 10-digit phone number"
                      maxLength="10"
                      className={formErrors.phone ? 'error' : ''}
                    />
                    {formErrors.phone && <span className="field-error">{formErrors.phone}</span>}
                  </div>
                  <div className="form-group">
                    <label>Email Address *</label>
                    <input
                      type="email"
                      name="email"
                      value={patientInfo.email}
                      onChange={handlePatientInfoChange}
                      placeholder="Enter email address"
                      className={formErrors.email ? 'error' : ''}
                    />
                    {formErrors.email && <span className="field-error">{formErrors.email}</span>}
                  </div>
                  <div className="form-group full-width">
                    <label>Address</label>
                    <input
                      type="text"
                      name="address"
                      value={patientInfo.address}
                      onChange={handlePatientInfoChange}
                      placeholder="Enter address"
                    />
                  </div>
                </div>
              </div>

              <div className="form-section">
                <h3><File /> Medical Information</h3>
                <div className="form-grid">
                  <div className="form-group full-width">
                    <label>Reason for Examination</label>
                    <input
                      type="text"
                      name="reasonForExam"
                      value={patientInfo.reasonForExam}
                      onChange={handlePatientInfoChange}
                      placeholder="e.g., Routine screening, Follow-up, etc."
                    />
                  </div>
                  <div className="form-group">
                    <label>Previous Thermogram?</label>
                    <select
                      name="previousThermogram"
                      value={patientInfo.previousThermogram}
                      onChange={handlePatientInfoChange}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Family History of Breast Cancer?</label>
                    <select
                      name="familyHistory"
                      value={patientInfo.familyHistory}
                      onChange={handlePatientInfoChange}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                </div>
              </div>

              {error && (
                <div className="error-message">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              <div className="form-actions">
                <Link to="/history" className="btn btn-secondary">
                  <History />
                  View Previous Reports
                </Link>
                <button className="btn btn-primary btn-lg" onClick={proceedToUpload}>
                  Proceed to Upload
                  <Upload size={18} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Image Upload */}
        {step === 2 && (
          <div className="upload-section">
            <div className="section-header">
              <h1>Upload Thermal Image</h1>
              <p>Upload a thermal breast image for AI-powered analysis</p>
            </div>

            <div className="upload-content">
              <div className="upload-zone-container">
                <div 
                  className={`upload-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={handleUploadClick}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".png,.jpg,.jpeg"
                    onChange={handleInputChange}
                    className="file-input"
                    onClick={(e) => e.stopPropagation()}
                  />
                  
                  {previewUrl ? (
                    <div className="preview-container">
                      <img src={previewUrl} alt="Preview" className="image-preview" />
                      <div className="preview-overlay">
                        <RotateCcw size={24} />
                        <span>Click to change</span>
                      </div>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <Upload className="upload-icon" />
                      <h3>Drag and drop your thermal image here</h3>
                      <p>or click to browse</p>
                      <span className="file-types">Supports PNG and JPEG formats</span>
                    </div>
                  )}
                </div>

                {error && (
                  <div className="error-message">
                    <AlertCircle size={18} />
                    <span>{error}</span>
                  </div>
                )}

                <div className="patient-info-summary">
                  <h4><User /> Patient: {patientInfo.firstName} {patientInfo.lastName}</h4>
                </div>

                <div className="upload-actions">
                  <button className="btn btn-secondary" onClick={() => setStep(1)}>
                    Back
                  </button>
                  <button
                    className="btn btn-primary btn-lg"
                    onClick={analyzeImage}
                    disabled={!selectedFile || isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader className="spinner" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Thermometer size={18} />
                        Analyze Thermal Image
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="upload-info">
                <h3><Info /> About Thermal Imaging</h3>
                <ul>
                  <li>Thermal imaging detects heat patterns in breast tissue</li>
                  <li>Cancerous tissue often shows higher temperature due to increased blood flow</li>
                  <li>This is a non-invasive, radiation-free screening method</li>
                  <li>Best results when images are taken in a temperature-controlled environment</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="results-section">
            <div className="section-header">
              <h1>Analysis Results</h1>
              <p>AI-powered thermal image analysis report</p>
            </div>

            <div className="results-content">
              <div className="result-card">
                <div className={`result-header ${getResultClass(result.prediction)}`}>
                  {getResultIcon(result.prediction)}
                  <div className="result-title">
                    <h2>{result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1)}</h2>
                    <span className="confidence">Confidence: {result.confidence}%</span>
                  </div>
                </div>

                <div className="result-details">
                  <div className="probability-bars">
                    <h3>Classification Probabilities</h3>
                    <div className="probability-item">
                      <span className="label">Healthy</span>
                      <div className="bar-container">
                        <div 
                          className="bar healthy" 
                          style={{ width: `${result.probabilities.Healthy}%` }}
                        ></div>
                      </div>
                      <span className="value">{result.probabilities.Healthy}%</span>
                    </div>
                    <div className="probability-item">
                      <span className="label">Sick</span>
                      <div className="bar-container">
                        <div 
                          className="bar sick" 
                          style={{ width: `${result.probabilities.Sick}%` }}
                        ></div>
                      </div>
                      <span className="value">{result.probabilities.Sick}%</span>
                    </div>
                  </div>

                  <div className="patient-summary">
                    <h3>Patient Summary</h3>
                    <div className="summary-grid">
                      <div className="summary-item">
                        <User size={16} />
                        <span>{patientInfo.firstName} {patientInfo.lastName}</span>
                      </div>
                      <div className="summary-item">
                        <Calendar size={16} />
                        <span>DOB: {patientInfo.dateOfBirth}</span>
                      </div>
                      <div className="summary-item">
                        <Thermometer size={16} />
                        <span>Thermal Imaging Scan</span>
                      </div>
                      <div className="summary-item">
                        <Calendar size={16} />
                        <span>Analysis: {new Date(result.analysisDate).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>

                  {result.prediction.toLowerCase() === 'sick' && (
                    <div className="warning-box">
                      <AlertCircle size={20} />
                      <div>
                        <strong>Important Notice</strong>
                        <p>The AI analysis indicates potential abnormalities. Please consult a healthcare professional for further evaluation and diagnostic procedures.</p>
                      </div>
                    </div>
                  )}

                  {result.prediction.toLowerCase() === 'healthy' && (
                    <div className="success-box">
                      <CheckCircle size={20} />
                      <div>
                        <strong>Normal Result</strong>
                        <p>The thermal image analysis shows no significant abnormalities. Continue with regular screening as recommended by your healthcare provider.</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="result-actions">
                  <button className="btn btn-secondary" onClick={resetAnalysis}>
                    <RotateCcw size={18} />
                    New Analysis
                  </button>
                  <button className="btn btn-primary" onClick={downloadReport}>
                    <Download size={18} />
                    Download Report
                  </button>
                </div>
              </div>

              <div className="disclaimer">
                <Info size={16} />
                <p>This AI analysis is for screening purposes only and should not replace professional medical diagnosis. Always consult with qualified healthcare providers for medical decisions.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ThermalAnalysis;
