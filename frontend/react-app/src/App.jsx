import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Analysis from './pages/Analysis';
import ThermalAnalysis from './pages/ThermalAnalysis';
import About from './pages/About';
import Contact from './pages/Contact';
import History from './pages/History';
import Patients from './pages/Patients';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/thermal-analysis" element={<ThermalAnalysis />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/history" element={<History />} />
            <Route path="/patients" element={<Patients />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
