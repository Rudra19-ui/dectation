import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Activity, ChevronDown, Thermometer } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showAnalysisDropdown, setShowAnalysisDropdown] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMenu = () => setIsOpen(!isOpen);

  const navLinks = [
    { path: '/', label: 'Home' },
    { path: '/patients', label: 'Patients' },
    { path: '/about', label: 'About' },
    { path: '/contact', label: 'Contact' },
  ];

  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <Activity className="logo-icon" />
          <span className="logo-text">
            <span className="logo-primary">Breast Cancer</span>
            <span className="logo-secondary">Detection AI</span>
          </span>
        </Link>

        <div className={`navbar-menu ${isOpen ? 'active' : ''}`}>
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`navbar-link ${location.pathname === link.path ? 'active' : ''}`}
              onClick={() => setIsOpen(false)}
            >
              {link.label}
            </Link>
          ))}
          
          {/* Analysis Dropdown */}
          <div 
            className="navbar-dropdown"
            onMouseEnter={() => setShowAnalysisDropdown(true)}
            onMouseLeave={() => setShowAnalysisDropdown(false)}
          >
            <Link
              to="/analysis"
              className={`navbar-link dropdown-toggle ${location.pathname.includes('/analysis') || location.pathname.includes('/thermal') ? 'active' : ''}`}
            >
              Analysis
              <ChevronDown size={16} className={`dropdown-arrow ${showAnalysisDropdown ? 'rotated' : ''}`} />
            </Link>
            {showAnalysisDropdown && (
              <div className="dropdown-menu">
                <Link
                  to="/analysis"
                  className={`dropdown-item ${location.pathname === '/analysis' ? 'active' : ''}`}
                  onClick={() => setIsOpen(false)}
                >
                  <Activity size={16} />
                  Mammography
                </Link>
                <Link
                  to="/thermal-analysis"
                  className={`dropdown-item ${location.pathname === '/thermal-analysis' ? 'active' : ''}`}
                  onClick={() => setIsOpen(false)}
                >
                  <Thermometer size={16} />
                  Thermal Imaging
                </Link>
              </div>
            )}
          </div>
          
          <Link to="/analysis" className="btn btn-primary navbar-cta" onClick={() => setIsOpen(false)}>
            Get Started
          </Link>
        </div>

        <button className="navbar-toggle" onClick={toggleMenu} aria-label="Toggle menu">
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </nav>
  );
};

export default Navbar;