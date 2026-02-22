import { useState, useEffect } from 'react';
import { 
  FileText, Search, Calendar, User, Download, 
  Eye, AlertCircle, Clock, Trash2
} from 'lucide-react';
import { API_URL } from '../config/api';
import './History.css';

const History = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/reports`);
      const data = await response.json();
      if (data.reports) {
        setReports(data.reports);
      } else {
        setError('Failed to load reports');
      }
    } catch (err) {
      setError('Unable to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
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

  return (
    <div className="history-page">
      <div className="container">
        <div className="page-header">
          <h1>Report History</h1>
          <p>View and manage previously generated reports</p>
        </div>

        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading reports...</p>
          </div>
        ) : error ? (
          <div className="error-state">
            <AlertCircle size={48} />
            <p>{error}</p>
            <button className="btn btn-primary" onClick={fetchReports}>
              Try Again
            </button>
          </div>
        ) : reports.length === 0 ? (
          <div className="empty-state">
            <FileText size={64} />
            <h3>No Reports Yet</h3>
            <p>Complete an analysis to generate your first report.</p>
            <a href="/analysis" className="btn btn-primary">
              Start Analysis
            </a>
          </div>
        ) : (
          <div className="reports-content">
            <div className="reports-list">
              <div className="list-header">
                <span className="count">{reports.length} Report{reports.length !== 1 ? 's' : ''}</span>
              </div>
              
              {reports.map((report) => (
                <div 
                  key={report.report_id} 
                  className={`report-item ${selectedReport?.report_id === report.report_id ? 'selected' : ''}`}
                  onClick={() => setSelectedReport(report)}
                >
                  <div className="report-icon">
                    <FileText size={20} />
                  </div>
                  <div className="report-info">
                    <div className="report-patient">
                      {report.patient?.firstName} {report.patient?.lastName}
                    </div>
                    <div className="report-meta">
                      <span><Calendar size={12} /> {formatDate(report.analysis_date)}</span>
                    </div>
                  </div>
                  <div className={`report-prediction ${getPredictionClass(report.prediction)}`}>
                    {report.prediction}
                  </div>
                </div>
              ))}
            </div>

            <div className="report-details">
              {selectedReport ? (
                <>
                  <div className="details-header">
                    <h2>Report Details</h2>
                    <span className="report-id">#{selectedReport.report_id}</span>
                  </div>

                  <div className="details-section">
                    <h3><User /> Patient Information</h3>
                    <div className="details-grid">
                      <div className="detail-item">
                        <span className="label">Name</span>
                        <span className="value">
                          {selectedReport.patient?.firstName} {selectedReport.patient?.lastName}
                        </span>
                      </div>
                      <div className="detail-item">
                        <span className="label">Date of Birth</span>
                        <span className="value">{selectedReport.patient?.dateOfBirth || 'N/A'}</span>
                      </div>
                      <div className="detail-item">
                        <span className="label">Gender</span>
                        <span className="value">{selectedReport.patient?.gender || 'N/A'}</span>
                      </div>
                      <div className="detail-item">
                        <span className="label">Phone</span>
                        <span className="value">{selectedReport.patient?.phone || 'N/A'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="details-section">
                    <h3>Analysis Results</h3>
                    <div className="result-box">
                      <div className={`main-result ${getPredictionClass(selectedReport.prediction)}`}>
                        <span className="prediction-label">Prediction</span>
                        <span className="prediction-value">{selectedReport.prediction}</span>
                      </div>
                      <div className="confidence-box">
                        <span className="confidence-label">Confidence</span>
                        <span className="confidence-value">{selectedReport.confidence?.toFixed(2)}%</span>
                      </div>
                    </div>

                    <div className="probabilities">
                      <h4>Probability Distribution</h4>
                      {selectedReport.probabilities && Object.entries(selectedReport.probabilities).map(([label, value]) => (
                        <div key={label} className="prob-item">
                          <span className="prob-label">{label}</span>
                          <div className="prob-bar">
                            <div 
                              className={`prob-fill ${label.toLowerCase()}`}
                              style={{ width: `${value}%` }}
                            ></div>
                          </div>
                          <span className="prob-value">{value}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="details-section">
                    <h3><Clock /> Analysis Date</h3>
                    <p className="date-value">{formatDate(selectedReport.analysis_date)}</p>
                  </div>

                  <div className="details-actions">
                    <button className="btn btn-primary">
                      <Download size={16} />
                      Download Report
                    </button>
                  </div>
                </>
              ) : (
                <div className="no-selection">
                  <Eye size={48} />
                  <p>Select a report to view details</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;