@echo off
echo ========================================
echo Breast Cancer Classification Setup
echo ========================================
echo.

echo Creating virtual environment...
python -m venv breast_cancer_env
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    echo Please ensure Python is installed and in PATH
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call breast_cancer_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
echo Installing TensorFlow...
pip install tensorflow

echo Installing OpenCV...
pip install opencv-python

echo Installing scikit-learn...
pip install scikit-learn

echo Installing matplotlib...
pip install matplotlib

echo Installing numpy...
pip install numpy

echo Installing additional packages for the project...
pip install pillow
pip install seaborn
pip install pydicom

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future:
echo   breast_cancer_env\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
echo Virtual environment created in: breast_cancer_env
echo.
pause 