import { useState, useRef } from 'react';
import { 
  Upload, Image, FileText, AlertCircle, CheckCircle, 
  Loader, Download, RotateCcw, Info, User, Calendar, 
  Phone, Mail, File, History
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { API_URL } from '../config/api';
import './Analysis.css';

const Analysis = () => {
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
    previousMammogram: 'No',
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
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom'];
      if (!validTypes.includes(file.type) && !file.name.endsWith('.dcm')) {
        setError('Please upload a valid image file (PNG, JPEG, or DICOM)');
        return;
      }

      // Check if this looks like a thermal image
      const fileName = file.name.toLowerCase();
      if (fileName.includes('thermal') || 
          fileName.includes('thermography') ||
          fileName.includes('infrared') ||
          fileName.includes('ir_')) {
        setError('This appears to be a thermal image. Please upload a mammography image instead, or use the Thermal Imaging Analysis page.');
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

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }
      
      // Parse JSON response from backend
      const data = await response.json();
      console.log('Prediction from backend:', data);
      
      // Extract prediction data
      const prediction = data.prediction || 'Normal';
      const confidence = data.confidence ? Math.round(data.confidence * 100) : 75;
      
      // Build probabilities object from arrays
      const probabilities = {
        Normal: data.probabilities ? Math.round(data.probabilities[2] * 100) : 0,
        Benign: data.probabilities ? Math.round(data.probabilities[0] * 100) : 0,
        Malignant: data.probabilities ? Math.round(data.probabilities[1] * 100) : 0
      };

      const analysisResults = {
        prediction: prediction.charAt(0).toUpperCase() + prediction.slice(1), // Capitalize
        confidence: confidence,
        probabilities: probabilities,
        patientInfo: patientInfo,
        analysisDate: new Date().toISOString()
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
    // First, save the report and generate PDF
    try {
      console.log('Sending report data:', result);
      
      const response = await fetch(`${API_URL}/generate-report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({...result, scanType: 'mammography'}),
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      if (data.success && data.pdf_url) {
        // Fetch the PDF as a blob and download it
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
        link.download = `BC_Report_${patientInfo.lastName}_${new Date().toISOString().split('T')[0]}.pdf`;
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
      previousMammogram: 'No',
      familyHistory: 'No'
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getResultClass = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'normal': return 'result-normal';
      case 'benign': return 'result-benign';
      case 'malignant': return 'result-malignant';
      default: return '';
    }
  };

  const getResultIcon = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'normal': return <CheckCircle className="result-icon success" />;
      case 'benign': return <Info className="result-icon warning" />;
      case 'malignant': return <AlertCircle className="result-icon danger" />;
      default: return null;
    }
  };

  return (
    <div className="analysis-page">
      <div className="container">
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
              <p>Please enter patient details before proceeding with the analysis</p>
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
                    <label>Previous Mammogram?</label>
                    <select
                      name="previousMammogram"
                      value={patientInfo.previousMammogram}
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
              <h1>Upload Mammogram</h1>
              <p>Upload a mammogram image for AI-powered analysis</p>
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
                    accept=".png,.jpg,.jpeg,.dcm"
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
                      <h3>Drag and drop your image here</h3>
                      <p>or click to browse</p>
                      <span className="file-types">Supports PNG, JPEG, and DICOM formats</span>
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
                        <Image />
                        Analyze Image
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="results-section">
            <div className="section-header">
              <h1>Analysis Results</h1>
              <p>Results for {patientInfo.firstName} {patientInfo.lastName}</p>
            </div>

            <div className="results-content">
              <div className="results-grid">
                {/* Patient Info Card */}
                <div className="patient-card">
                  <h3><User /> Patient Information</h3>
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="label">Name</span>
                      <span className="value">{patientInfo.firstName} {patientInfo.lastName}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Date of Birth</span>
                      <span className="value">{patientInfo.dateOfBirth}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Gender</span>
                      <span className="value">{patientInfo.gender}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Analysis Date</span>
                      <span className="value">{new Date(result.analysisDate).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>

                {/* Result Card */}
                <div className={`result-card ${getResultClass(result.prediction)}`}>
                  <div className="result-header">
                    {getResultIcon(result.prediction)}
                    <div className="result-info">
                      <h3>Analysis Result</h3>
                      <span className={`result-badge ${result.prediction?.toLowerCase()}`}>
                        {result.prediction}
                      </span>
                    </div>
                  </div>
                  
                  <div className="confidence-display">
                    <div className="confidence-label">
                      <span>Confidence Score</span>
                      <span className="confidence-value">{result.confidence}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Probabilities Card */}
                <div className="probabilities-card">
                  <h4>Probability Distribution</h4>
                  <div className="probabilities-list">
                    {result.probabilities && Object.entries(result.probabilities).map(([label, value]) => (
                      <div key={label} className="probability-item">
                        <span className="probability-label">{label}</span>
                        <div className="probability-bar-container">
                          <div 
                            className={`probability-bar ${label.toLowerCase()}`}
                            style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                          ></div>
                        </div>
                        <span className="probability-value">{typeof value === 'number' ? value.toFixed(2) : value}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Info Cards */}
              <div className="info-cards">
                <div className="info-card normal">
                  <h5>Normal</h5>
                  <p>No signs of cancer detected. Regular screening recommended.</p>
                </div>
                <div className="info-card benign">
                  <h5>Benign</h5>
                  <p>Non-cancerous abnormality. Follow-up recommended.</p>
                </div>
                <div className="info-card malignant">
                  <h5>Malignant</h5>
                  <p>Potential cancer detected. Immediate consultation advised.</p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="result-actions">
                <button className="btn btn-secondary" onClick={resetAnalysis}>
                  <RotateCcw />
                  New Analysis
                </button>
                <button className="btn btn-accent btn-lg" onClick={downloadReport}>
                  <Download />
                  Download Report
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="disclaimer">
          <AlertCircle size={18} />
          <p>
            <strong>Disclaimer:</strong> This AI tool is for educational and research purposes only. 
            It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with a qualified healthcare provider for medical decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Analysis;