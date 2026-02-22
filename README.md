# Breast Cancer Detection AI

A deep learning-based breast cancer detection system using PyTorch and Flask.

## Project Structure

The project has been organized into the following structure:

```
project/
├── backend/                    # Backend application
│   ├── api/                    # Flask API endpoints
│   ├── auth/                   # Authentication and authorization (RBAC)
│   ├── config/                 # Configuration files
│   ├── data/                   # Data files (CSV, JSON)
│   ├── datasets/               # Dataset storage
│   │   ├── organized_dataset/  # Organized image dataset
│   │   ├── preprocessed_dataset/ # Preprocessed images
│   │   └── split_dataset/      # Train/val/test splits
│   ├── evaluation/             # Model evaluation scripts
│   ├── model_files/            # Additional model files
│   ├── models/                 # Model definitions and saved models
│   ├── notebooks/              # Jupyter notebooks
│   ├── prediction/             # Prediction scripts
│   ├── reports/                # Generated reports
│   ├── routes/                 # API route handlers
│   ├── scripts/                # Utility scripts
│   ├── src/                    # Source code modules
│   ├── training/               # Training scripts
│   └── utils/                  # Utility functions
│
├── frontend/                   # Frontend application
│   ├── templates/              # HTML templates
│   ├── webapp/                 # Streamlit web applications
│   ├── frontend.html           # Frontend HTML
│   └── image_upload_app.py     # Image upload application
│
├── docs/                       # Documentation files
│   └── *.md                    # Various guides and documentation
│
├── scripts/                    # Setup and run scripts
│   ├── run_all.bat            # Run all components
│   ├── run_project.bat        # Run project (Windows)
│   ├── run_project.ps1        # Run project (PowerShell)
│   ├── setup_environment.bat  # Setup environment (Windows)
│   └── setup_environment.ps1  # Setup environment (PowerShell)
│
├── .github/                    # GitHub configuration
├── .pre-commit-config.yaml     # Pre-commit hooks
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── requirements_enhanced.txt   # Enhanced dependencies
└── README.md                   # This file
```

## Features

- **Deep Learning Model**: ResNet50-based breast cancer detection
- **Multi-class Classification**: Normal, Benign, Malignant
- **Web Interface**: Streamlit-based frontend for easy interaction
- **REST API**: Flask-based API for integration
- **DICOM Support**: Convert DICOM images to PNG
- **Grad-CAM Visualization**: Explainable AI for predictions
- **Report Generation**: Automated medical report generation

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Flask
- Streamlit

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend API:
```bash
cd backend
python api/app.py
```

4. Run the frontend (Streamlit):
```bash
cd frontend
streamlit run webapp/streamlit_app.py
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload image for prediction

## Model Training

Training scripts are located in `backend/training/`:

```bash
python backend/training/train_model.py
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Complete Project Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md)
- [Training Guide](docs/TRAINING_README.md)
- [Prediction Guide](docs/PREDICTION_GUIDE.md)
- [Setup Instructions](docs/SETUP_README.md)

## License

This project is for educational and research purposes.

## Contributing

Please read the documentation before contributing to this project.
