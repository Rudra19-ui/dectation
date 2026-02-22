// API Configuration
// Change this URL if the backend server runs on a different port
const API_CONFIG = {
  // Default backend URL - change port if needed
  BASE_URL: 'http://localhost:8000',
  
  // API endpoints
  ENDPOINTS: {
    PREDICT: '/predict',
    PREDICT_THERMAL: '/predict-thermal',
    GENERATE_REPORT: '/generate-report',
    REPORTS: '/reports',
    PATIENTS: '/patients',
    MODEL_INFO: '/model-info',
    THERMAL_MODEL_INFO: '/thermal-model-info',
  }
};

// Export the base URL for simple usage
export const API_URL = API_CONFIG.BASE_URL;

// Export full config for advanced usage
export default API_CONFIG;
