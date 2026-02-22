# Breast Cancer Classification Setup Script
# PowerShell version for Windows

Write-Host "========================================" -ForegroundColor Green
Write-Host "Breast Cancer Classification Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found. Please install Python and add it to PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv breast_cancer_env
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\breast_cancer_env\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow

$packages = @(
    "tensorflow",
    "opencv-python", 
    "scikit-learn",
    "matplotlib",
    "numpy",
    "pillow",
    "seaborn",
    "pydicom"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $package installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install $package" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future:" -ForegroundColor Yellow
Write-Host "  .\breast_cancer_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "Virtual environment created in: breast_cancer_env" -ForegroundColor Cyan
Write-Host ""

# Test the installation
Write-Host "Testing installation..." -ForegroundColor Yellow
try {
    python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"
    python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"
    python -c "import matplotlib; print('matplotlib version:', matplotlib.__version__)"
    python -c "import numpy; print('NumPy version:', numpy.__version__)"
    Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Some packages may not be working correctly" -ForegroundColor Red
}

Read-Host "Press Enter to exit" 