import { 
  Brain, Cpu, Database, Shield, Target, Users, 
  Award, BookOpen, CheckCircle, ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import './About.css';

const About = () => {
  const technologies = [
    { icon: <Brain />, name: 'PyTorch', description: 'Deep learning framework for model training' },
    { icon: <Cpu />, name: 'ResNet50', description: 'Pre-trained CNN architecture' },
    { icon: <Database />, name: 'Flask', description: 'REST API backend for model serving' },
    { icon: <Shield />, name: 'Security', description: 'HIPAA compliant data handling' }
  ];

  const milestones = [
    { title: 'Research & Planning', description: 'Literature review, dataset collection, architecture design' },
    { title: 'Development', description: 'Model training, API development, frontend design' },
    { title: 'Testing & Validation', description: 'Performance evaluation, clinical validation' },
    { title: 'Deployment', description: 'Production deployment, documentation, release' }
  ];

  const features = [
    'Completed comprehensive literature review on breast cancer detection',
    'Collected and preprocessed 1,198 mammogram images',
    'Achieved 82.87% accuracy on test set',
    'Implemented Grad-CAM for explainable AI',
    'Developed REST API for model serving',
    'Created professional web interface',
    'Added PDF report generation',
    'Integrated DICOM support'
  ];

  return (
    <div className="about-page">
      {/* Hero Section */}
      <section className="about-hero">
        <div className="container">
          <div className="hero-content">
            <h1>About Our Project</h1>
            <p>Advancing breast cancer detection through artificial intelligence</p>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="mission-section">
        <div className="container">
          <div className="mission-card">
            <h2>Our Mission</h2>
            <p>
              Our mission is to leverage cutting-edge artificial intelligence technology to assist healthcare 
              professionals in the early detection and diagnosis of breast cancer. We believe that AI can be a 
              powerful tool in the fight against cancer, helping to improve accuracy, reduce diagnostic time, 
              and ultimately save lives.
            </p>
            <p>
              By combining deep learning expertise with medical imaging knowledge, we've developed a system 
              that can classify mammogram images into Normal, Benign, and Malignant categories with high accuracy, 
              while providing explainable insights through Grad-CAM visualization.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">82.87%</div>
              <div className="stat-label">Accuracy</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">79.24%</div>
              <div className="stat-label">F1 Score</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">1,198</div>
              <div className="stat-label">Training Images</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">3</div>
              <div className="stat-label">Classification Classes</div>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="technology-section">
        <div className="container">
          <div className="section-header">
            <h2>Technology Stack</h2>
            <p>Built with state-of-the-art technologies</p>
          </div>
          <div className="tech-grid">
            {technologies.map((tech, index) => (
              <div key={index} className="tech-card">
                <div className="tech-icon">{tech.icon}</div>
                <h3>{tech.name}</h3>
                <p>{tech.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Model Architecture Section */}
      <section className="architecture-section">
        <div className="container">
          <div className="section-header">
            <h2>Model Architecture</h2>
            <p>Deep learning powered classification</p>
          </div>
          <div className="architecture-grid">
            <div className="architecture-card">
              <h3><Target /> ResNet50 Architecture</h3>
              <p>
                Our model is built on ResNet50, a 50-layer deep convolutional neural network that has 
                revolutionized image classification.
              </p>
              <ul className="feature-list">
                <li><CheckCircle size={16} /> Pre-trained on ImageNet (1.2M images)</li>
                <li><CheckCircle size={16} /> Residual connections for deep network training</li>
                <li><CheckCircle size={16} /> Transfer learning for medical imaging</li>
                <li><CheckCircle size={16} /> Fine-tuned on combined BUSI dataset</li>
                <li><CheckCircle size={16} /> Custom classifier head for 3-class output</li>
              </ul>
            </div>
            <div className="architecture-card">
              <h3><BookOpen /> Training Details</h3>
              <p>
                The model was trained with careful attention to medical imaging requirements:
              </p>
              <ul className="feature-list">
                <li><CheckCircle size={16} /> 838 training samples with augmentation</li>
                <li><CheckCircle size={16} /> Weighted sampling for class balance</li>
                <li><CheckCircle size={16} /> Early stopping to prevent overfitting</li>
                <li><CheckCircle size={16} /> Learning rate scheduling</li>
                <li><CheckCircle size={16} /> Cross-entropy loss with class weights</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Section */}
      <section className="dataset-section">
        <div className="container">
          <div className="dataset-card">
            <h2>Dataset Information</h2>
            <h3>BUSI Dataset (Breast Ultrasound Images)</h3>
            <p>
              The model was trained on the Breast Ultrasound Images (BUSI) dataset, a comprehensive 
              collection of breast ultrasound images with expert annotations.
            </p>
            <div className="dataset-stats">
              <div className="dataset-stat normal">
                <div className="stat-number">133</div>
                <div className="stat-name">Normal</div>
              </div>
              <div className="dataset-stat benign">
                <div className="stat-number">437</div>
                <div className="stat-name">Benign</div>
              </div>
              <div className="dataset-stat malignant">
                <div className="stat-number">210</div>
                <div className="stat-name">Malignant</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="timeline-section">
        <div className="container">
          <div className="section-header">
            <h2>Project Timeline</h2>
            <p>Our development journey</p>
          </div>
          <div className="timeline-grid">
            <div className="timeline-items">
              {milestones.map((milestone, index) => (
                <div key={index} className="timeline-item">
                  <div className="timeline-marker">{index + 1}</div>
                  <div className="timeline-content">
                    <h4>{milestone.title}</h4>
                    <p>{milestone.description}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="milestones-card">
              <h3>Key Milestones</h3>
              <ul className="milestones-list">
                {features.map((feature, index) => (
                  <li key={index}>
                    <CheckCircle size={16} />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <h2>Ready to Try It?</h2>
            <p>Experience the power of AI-assisted breast cancer detection</p>
            <Link to="/analysis" className="btn btn-primary btn-lg">
              Start Analysis
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="disclaimer-section">
        <div className="container">
          <div className="disclaimer-card">
            <h4>Educational & Research Purpose</h4>
            <p>
              This project is developed for educational and research purposes. It is not intended to be used 
              as a medical diagnostic tool without proper clinical validation and regulatory approval. 
              Always consult with qualified healthcare professionals for medical decisions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;