import { useState, useEffect } from 'react';
import { 
  Users, Search, Calendar, FileText, Download, 
  Eye, AlertCircle, Plus, ChevronRight, Activity, RefreshCw
} from 'lucide-react';
import { API_URL } from '../config/api';
import './Patients.css';

const Patients = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPatient, setSelectedPatient] = useState(null);

  // Fetch patients on mount and when window gains focus
  useEffect(() => {
    fetchPatients();
    
    // Refresh data when window gains focus
    const handleFocus = () => fetchPatients();
    window.addEventListener('focus', handleFocus);
    
    return () => window.removeEventListener('focus', handleFocus);
  }, []);

  const fetchPatients = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('Fetching patients from:', `${API_URL}/patients`);
      const response = await fetch(`${API_URL}/patients`);
      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Patients data:', data);
      if (data.patients) {
        setPatients(data.patients);
        console.log('Patients loaded:', data.patients.length);
      } else {
        setError('Failed to load patients');
      }
    } catch (err) {
      console.error('Error fetching patients:', err);
      setError('Unable to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const fetchReports = async () => {
    try {
      const response = await fetch(`${API_URL}/reports`);
      const data = await response.json();
      return data.reports || [];
    } catch (err) {
      return [];
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getPredictionClass = (prediction) => {
    switch (prediction?.toLowerCase()) {
      case 'normal': return 'prediction-normal';
      case 'benign': return 'prediction-benign';
      case 'malignant': return 'prediction-malignant';
      default: return '';
    }
  };

  const downloadPDF = async (pdfPath) => {
    try {
      // Fetch the PDF as a blob and download it
      const response = await fetch(`${API_URL}/reports/${pdfPath}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = pdfPath;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    } catch (err) {
      console.error('Download failed:', err);
    }
  };

  const filteredPatients = patients.filter(patient => {
    const searchLower = searchTerm.toLowerCase();
    return (
      patient.firstName?.toLowerCase().includes(searchLower) ||
      patient.lastName?.toLowerCase().includes(searchLower) ||
      patient.email?.toLowerCase().includes(searchLower)
    );
  });

  return (
    <div className="patients-page">
      <div className="container">
        <div className="page-header">
          <div className="header-content">
            <h1><Users /> Patient Records</h1>
            <p>View and manage all patient records and their analysis history</p>
          </div>
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={fetchPatients} disabled={loading}>
              <RefreshCw size={18} className={loading ? 'spinning' : ''} />
              Refresh
            </button>
            <a href="/analysis" className="btn btn-primary">
              <Plus size={18} />
              New Analysis
            </a>
          </div>
        </div>

        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading patient records...</p>
          </div>
        ) : error ? (
          <div className="error-state">
            <AlertCircle size={48} />
            <p>{error}</p>
            <button className="btn btn-primary" onClick={fetchPatients}>
              Try Again
            </button>
          </div>
        ) : (
          <div className="patients-content">
            <div className="patients-sidebar">
              <div className="search-box">
                <Search size={18} />
                <input
                  type="text"
                  placeholder="Search patients..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              <div className="patients-list">
                {filteredPatients.length === 0 ? (
                  <div className="empty-list">
                    <Users size={32} />
                    <p>No patients found</p>
                  </div>
                ) : (
                  filteredPatients.map((patient, index) => (
                    <div 
                      key={index}
                      className={`patient-item ${selectedPatient === patient ? 'selected' : ''}`}
                      onClick={() => setSelectedPatient(patient)}
                    >
                      <div className="patient-avatar">
                        {patient.firstName?.charAt(0)}{patient.lastName?.charAt(0)}
                      </div>
                      <div className="patient-info">
                        <div className="patient-name">
                          {patient.firstName} {patient.lastName}
                        </div>
                        <div className="patient-meta">
                          <span>{patient.reports?.length || 0} reports</span>
                        </div>
                      </div>
                      <ChevronRight size={18} className="arrow" />
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="patient-details">
              {selectedPatient ? (
                <>
                  <div className="details-header">
                    <div className="patient-profile">
                      <div className="profile-avatar">
                        {selectedPatient.firstName?.charAt(0)}{selectedPatient.lastName?.charAt(0)}
                      </div>
                      <div className="profile-info">
                        <h2>{selectedPatient.firstName} {selectedPatient.lastName}</h2>
                        <p><Calendar size={14} /> DOB: {selectedPatient.dateOfBirth || 'N/A'}</p>
                        <p><Activity size={14} /> Gender: {selectedPatient.gender || 'N/A'}</p>
                      </div>
                    </div>
                    <div className="contact-info">
                      <p>{selectedPatient.email || 'No email'}</p>
                      <p>{selectedPatient.phone || 'No phone'}</p>
                    </div>
                  </div>

                  <div className="reports-section">
                    <h3><FileText /> Analysis History</h3>
                    
                    {selectedPatient.reports && selectedPatient.reports.length > 0 ? (
                      <div className="reports-table">
                        <div className="table-header">
                          <span>Date</span>
                          <span>Report ID</span>
                          <span>Prediction</span>
                          <span>Confidence</span>
                          <span>Actions</span>
                        </div>
                        {selectedPatient.reports.map((report, idx) => (
                          <div key={idx} className="table-row">
                            <span>{formatDate(report.analysis_date)}</span>
                            <span className="report-id">#{report.report_id}</span>
                            <span className={`prediction-badge ${getPredictionClass(report.prediction)}`}>
                              {report.prediction}
                            </span>
                            <span>{report.confidence?.toFixed(2)}%</span>
                            <div className="actions">
                              <button 
                                className="action-btn"
                                onClick={() => downloadPDF(report.pdf_path)}
                                title="Download PDF"
                              >
                                <Download size={16} />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="no-reports">
                        <FileText size={32} />
                        <p>No analysis records found for this patient</p>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="no-selection">
                  <Users size={64} />
                  <h3>Select a Patient</h3>
                  <p>Choose a patient from the list to view their analysis history</p>
                </div>
              )}
            </div>
          </div>
        )}

        {patients.length === 0 && !loading && (
          <div className="empty-state">
            <Users size={64} />
            <h3>No Patient Records Yet</h3>
            <p>Complete your first analysis to create patient records.</p>
            <a href="/analysis" className="btn btn-primary">
              Start Analysis
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default Patients;