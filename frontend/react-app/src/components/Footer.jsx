import { Link } from 'react-router-dom';
import { Activity, Mail, Phone, MapPin, Linkedin, Twitter, Github, Youtube } from 'lucide-react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-grid">
          <div className="footer-brand">
            <Link to="/" className="footer-logo">
              <Activity className="footer-logo-icon" />
              <span className="footer-logo-text">
                <span>Breast Cancer</span>
                <span>Detection AI</span>
              </span>
            </Link>
            <p className="footer-description">
              Advanced AI-powered breast cancer detection system helping healthcare professionals 
              make accurate diagnoses with deep learning technology.
            </p>
            <div className="footer-social">
              <a href="#" className="social-link" aria-label="LinkedIn"><Linkedin size={18} /></a>
              <a href="#" className="social-link" aria-label="Twitter"><Twitter size={18} /></a>
              <a href="#" className="social-link" aria-label="Github"><Github size={18} /></a>
              <a href="#" className="social-link" aria-label="Youtube"><Youtube size={18} /></a>
            </div>
          </div>

          <div className="footer-links">
            <h4>Quick Links</h4>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><Link to="/analysis">Analysis</Link></li>
              <li><Link to="/about">About Us</Link></li>
              <li><Link to="/contact">Contact</Link></li>
            </ul>
          </div>

          <div className="footer-links">
            <h4>Resources</h4>
            <ul>
              <li><a href="#">Documentation</a></li>
              <li><a href="#">API Reference</a></li>
              <li><a href="#">Research Papers</a></li>
              <li><a href="#">FAQ</a></li>
            </ul>
          </div>

          <div className="footer-contact">
            <h4>Contact Us</h4>
            <div className="contact-item">
              <Mail size={16} />
              <span>support@bcdetection.ai</span>
            </div>
            <div className="contact-item">
              <Phone size={16} />
              <span>+1 (555) 123-4567</span>
            </div>
            <div className="contact-item">
              <MapPin size={16} />
              <span>Boston, MA 02101</span>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Breast Cancer Detection AI. All rights reserved.</p>
          <div className="footer-bottom-links">
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
            <a href="#">Cookie Policy</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;