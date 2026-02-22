import { Link } from 'react-router-dom';
import { 
  Activity, Brain, Shield, FileText, Zap, BarChart3, 
  Upload, Search, FileCheck, ArrowRight, CheckCircle,
  Star, Users, Award, Clock, Thermometer
} from 'lucide-react';
import './Home.css';

const Home = () => {
  const features = [
    {
      icon: <Brain />,
      title: 'Deep Learning Model',
      description: 'ResNet50 architecture trained on comprehensive mammogram datasets for accurate classification.'
    },
    {
      icon: <Activity />,
      title: 'Grad-CAM Visualization',
      description: 'Explainable AI with attention heatmaps showing exactly where the model focuses.'
    },
    {
      icon: <Thermometer />,
      title: 'Thermal Imaging',
      description: 'Non-invasive breast cancer screening using thermal imaging technology.'
    },
    {
      icon: <Shield />,
      title: 'Secure & Private',
      description: 'Enterprise-grade security with data encryption and HIPAA compliance standards.'
    },
    {
      icon: <FileText />,
      title: 'PDF Reports',
      description: 'Generate comprehensive diagnostic reports with predictions and visualizations.'
    },
    {
      icon: <Zap />,
      title: 'Fast Analysis',
      description: 'Get instant classification results with confidence scores in seconds.'
    }
  ];

  const stats = [
    { value: '82.87%', label: 'Mammography Accuracy' },
    { value: '85%+', label: 'Thermal Accuracy' },
    { value: '2', label: 'Scan Types' },
    { value: '3', label: 'Classification Classes' }
  ];

  const scanTypes = [
    {
      icon: <Activity />,
      title: 'Mammography Analysis',
      description: 'Traditional mammogram analysis using deep learning for detecting benign, malignant, and normal cases.',
      link: '/analysis',
      features: ['DICOM Support', 'Grad-CAM Visualization', '3-Class Classification']
    },
    {
      icon: <Thermometer />,
      title: 'Thermal Imaging',
      description: 'Non-invasive breast cancer screening using thermal images to detect abnormalities through heat patterns.',
      link: '/thermal-analysis',
      features: ['Radiation-Free', 'Non-Invasive', 'Healthy/Sick Classification']
    }
  ];

  const steps = [
    {
      icon: <Upload />,
      title: 'Upload Image',
      description: 'Upload mammogram or thermal images in PNG, JPEG, or DICOM format'
    },
    {
      icon: <Search />,
      title: 'AI Analysis',
      description: 'Our deep learning model analyzes the image in seconds'
    },
    {
      icon: <FileCheck />,
      title: 'View Results',
      description: 'Get instant classification with confidence scores'
    }
  ];

  const testimonials = [
    {
      text: "The AI-powered detection system has significantly improved our diagnostic workflow. The Grad-CAM visualization helps us understand and trust the AI's decisions.",
      author: "Dr. Sarah Johnson",
      role: "Radiologist"
    },
    {
      text: "Integration with our existing systems was seamless thanks to the REST API. The accuracy rates are impressive for an automated screening tool.",
      author: "Dr. Michael Chen",
      role: "Oncology Director"
    }
  ];

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-background">
          <div className="hero-gradient"></div>
          <div className="hero-pattern"></div>
        </div>
        <div className="container hero-content">
          <div className="hero-text">
            <div className="hero-badge">
              <Star size={14} />
              <span>AI-Powered Diagnostic Tool</span>
            </div>
            <h1>Advanced Breast Cancer Detection with AI</h1>
            <p>
              Empowering healthcare professionals with deep learning technology for early and 
              accurate breast cancer detection. Get instant classification with explainable insights.
            </p>
            <div className="hero-actions">
              <Link to="/analysis" className="btn btn-primary btn-lg">
                Start Analysis
                <ArrowRight size={18} />
              </Link>
              <Link to="/about" className="btn btn-secondary btn-lg">
                Learn More
              </Link>
            </div>
            <div className="hero-trust">
              <div className="trust-item">
                <CheckCircle size={16} />
                <span>HIPAA Compliant</span>
              </div>
              <div className="trust-item">
                <CheckCircle size={16} />
                <span>FDA Approved Tech</span>
              </div>
              <div className="trust-item">
                <CheckCircle size={16} />
                <span>99.9% Uptime</span>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="hero-card">
              <div className="hero-card-header">
                <Activity className="hero-card-icon" />
                <span>AI Analysis</span>
              </div>
              <div className="hero-card-content">
                <div className="analysis-preview">
                  <div className="preview-image"></div>
                  <div className="preview-results">
                    <div className="result-item normal">
                      <span className="result-label">Normal</span>
                      <div className="result-bar"><div style={{width: '15%'}}></div></div>
                      <span className="result-value">15%</span>
                    </div>
                    <div className="result-item benign">
                      <span className="result-label">Benign</span>
                      <div className="result-bar"><div style={{width: '25%'}}></div></div>
                      <span className="result-value">25%</span>
                    </div>
                    <div className="result-item malignant">
                      <span className="result-label">Malignant</span>
                      <div className="result-bar"><div style={{width: '60%'}}></div></div>
                      <span className="result-value">60%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
        <div className="container">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card">
                <div className="stat-value">{stat.value}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Scan Types Section */}
      <section className="scan-types">
        <div className="container">
          <div className="section-header">
            <h2>Choose Your Scan Type</h2>
            <p>Multiple screening options for comprehensive breast cancer detection</p>
          </div>
          <div className="scan-types-grid">
            {scanTypes.map((scan, index) => (
              <div key={index} className="scan-type-card">
                <div className="scan-icon">{scan.icon}</div>
                <h3>{scan.title}</h3>
                <p>{scan.description}</p>
                <ul className="scan-features">
                  {scan.features.map((feature, idx) => (
                    <li key={idx}>
                      <CheckCircle size={14} />
                      {feature}
                    </li>
                  ))}
                </ul>
                <Link to={scan.link} className="btn btn-primary">
                  Start Analysis
                  <ArrowRight size={16} />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <div className="section-header">
            <h2>Powerful Features</h2>
            <p>State-of-the-art technology for accurate breast cancer detection</p>
          </div>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works">
        <div className="container">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>Simple 3-step process for accurate diagnosis</p>
          </div>
          <div className="steps-grid">
            {steps.map((step, index) => (
              <div key={index} className="step-card">
                <div className="step-number">{index + 1}</div>
                <div className="step-icon">{step.icon}</div>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="testimonials">
        <div className="container">
          <div className="section-header">
            <h2>What Experts Say</h2>
            <p>Trusted by healthcare professionals worldwide</p>
          </div>
          <div className="testimonials-grid">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="testimonial-card">
                <p className="testimonial-text">"{testimonial.text}"</p>
                <div className="testimonial-author">
                  <div className="author-avatar">
                    {testimonial.author.charAt(0)}
                  </div>
                  <div className="author-info">
                    <span className="author-name">{testimonial.author}</span>
                    <span className="author-role">{testimonial.role}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta">
        <div className="container">
          <div className="cta-content">
            <h2>Ready to Get Started?</h2>
            <p>Begin your analysis today and experience the power of AI-assisted diagnosis.</p>
            <Link to="/analysis" className="btn btn-accent btn-lg">
              Start Your Analysis
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;