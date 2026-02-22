import { useState } from 'react';
import { 
  Mail, Phone, MapPin, Clock, Send, 
  Linkedin, Twitter, Github, MessageCircle,
  CheckCircle, AlertCircle
} from 'lucide-react';
import './Contact.css';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: 'General Inquiry',
    message: ''
  });
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.email || !formData.message) {
      setError('Please fill in all required fields');
      return;
    }

    // Simulate form submission
    setSubmitted(true);
    setError('');
    setFormData({
      name: '',
      email: '',
      subject: 'General Inquiry',
      message: ''
    });
  };

  const contactInfo = [
    {
      icon: <Mail />,
      title: 'Email Us',
      value: 'support@bcdetection.ai',
      description: 'For general inquiries and support'
    },
    {
      icon: <Phone />,
      title: 'Call Us',
      value: '+1 (555) 123-4567',
      description: 'Mon-Fri from 9am to 6pm EST'
    },
    {
      icon: <MapPin />,
      title: 'Visit Us',
      value: 'Boston, MA 02101',
      description: 'Healthcare Innovation District'
    }
  ];

  const faqs = [
    {
      question: 'How accurate is the AI model?',
      answer: 'Our ResNet50-based model achieves 82.87% accuracy on the test dataset with a 79.24% F1 score. However, this tool is designed to assist, not replace, professional medical diagnosis.'
    },
    {
      question: 'What image formats are supported?',
      answer: 'We support PNG, JPEG, and DICOM formats. The system automatically preprocesses images to the required 224x224 resolution for analysis.'
    },
    {
      question: 'Is my data secure?',
      answer: 'Yes, we take data security seriously. All uploads are processed securely, and we do not store your images permanently. The system is designed with HIPAA compliance in mind.'
    },
    {
      question: 'Can I use this for clinical diagnosis?',
      answer: 'This tool is currently for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.'
    },
    {
      question: 'What is Grad-CAM visualization?',
      answer: 'Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainable AI technique that shows which regions of the image the model focused on for its prediction.'
    },
    {
      question: 'How can I collaborate on this project?',
      answer: 'We welcome collaborations! Please reach out through the contact form or email us directly at collaboration@bcdetection.ai with your proposal.'
    }
  ];

  return (
    <div className="contact-page">
      {/* Hero Section */}
      <section className="contact-hero">
        <div className="container">
          <div className="hero-content">
            <h1>Get In Touch</h1>
            <p>We're here to help and answer any questions you might have</p>
          </div>
        </div>
      </section>

      {/* Contact Cards */}
      <section className="contact-cards-section">
        <div className="container">
          <div className="contact-cards">
            {contactInfo.map((info, index) => (
              <div key={index} className="contact-card">
                <div className="contact-card-icon">{info.icon}</div>
                <h3>{info.title}</h3>
                <p className="contact-value">{info.value}</p>
                <p className="contact-description">{info.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Form and Info Section */}
      <section className="form-section">
        <div className="container">
          <div className="form-grid">
            {/* Contact Form */}
            <div className="form-container">
              <h2>Send Us a Message</h2>
              
              {submitted ? (
                <div className="success-message">
                  <CheckCircle size={24} />
                  <div>
                    <h4>Message Sent!</h4>
                    <p>Thank you for your message. We'll get back to you within 24-48 hours.</p>
                  </div>
                </div>
              ) : (
                <form onSubmit={handleSubmit}>
                  {error && (
                    <div className="error-message">
                      <AlertCircle size={18} />
                      <span>{error}</span>
                    </div>
                  )}

                  <div className="form-row">
                    <div className="form-group">
                      <label htmlFor="name">Full Name *</label>
                      <input
                        type="text"
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        placeholder="Enter your name"
                      />
                    </div>
                    <div className="form-group">
                      <label htmlFor="email">Email Address *</label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        placeholder="Enter your email"
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label htmlFor="subject">Subject</label>
                    <select
                      id="subject"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                    >
                      <option value="General Inquiry">General Inquiry</option>
                      <option value="Technical Support">Technical Support</option>
                      <option value="Collaboration Opportunity">Collaboration Opportunity</option>
                      <option value="Bug Report">Bug Report</option>
                      <option value="Feature Request">Feature Request</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="message">Message *</label>
                    <textarea
                      id="message"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      placeholder="Type your message here..."
                      rows={5}
                    />
                  </div>

                  <button type="submit" className="btn btn-primary btn-lg">
                    <Send size={18} />
                    Send Message
                  </button>
                </form>
              )}
            </div>

            {/* Info Panel */}
            <div className="info-panel">
              {/* Map Placeholder */}
              <div className="map-container">
                <div className="map-placeholder">
                  <MapPin size={48} />
                  <p>Interactive Map</p>
                </div>
              </div>

              {/* Office Hours */}
              <div className="office-hours">
                <h3><Clock /> Office Hours</h3>
                <div className="hours-list">
                  <div className="hours-item">
                    <span>Monday - Friday</span>
                    <span>9:00 AM - 6:00 PM</span>
                  </div>
                  <div className="hours-item">
                    <span>Saturday</span>
                    <span>10:00 AM - 2:00 PM</span>
                  </div>
                  <div className="hours-item">
                    <span>Sunday</span>
                    <span className="closed">Closed</span>
                  </div>
                </div>
              </div>

              {/* Social Links */}
              <div className="social-section">
                <h3>Connect With Us</h3>
                <div className="social-links">
                  <a href="#" className="social-link" aria-label="LinkedIn">
                    <Linkedin size={20} />
                  </a>
                  <a href="#" className="social-link" aria-label="Twitter">
                    <Twitter size={20} />
                  </a>
                  <a href="#" className="social-link" aria-label="Github">
                    <Github size={20} />
                  </a>
                  <a href="#" className="social-link" aria-label="Chat">
                    <MessageCircle size={20} />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="faq-section">
        <div className="container">
          <div className="section-header">
            <h2>Frequently Asked Questions</h2>
            <p>Find answers to common questions</p>
          </div>
          <div className="faq-grid">
            {faqs.map((faq, index) => (
              <div key={index} className="faq-item">
                <h4>{faq.question}</h4>
                <p>{faq.answer}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Support Banner */}
      <section className="support-section">
        <div className="container">
          <div className="support-banner">
            <h2>Need Technical Support?</h2>
            <p>Our team is ready to help you with any technical issues or questions.</p>
            <a href="mailto:support@bcdetection.ai" className="btn btn-accent btn-lg">
              <Mail size={18} />
              Contact Support Team
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Contact;