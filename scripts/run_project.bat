@echo off
echo ========================================
echo Breast Cancer Classification Project
echo ========================================
echo.

REM Set environment variables to avoid multiprocessing issues
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONPATH=.

echo Environment variables set:
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
echo MKL_NUM_THREADS=%MKL_NUM_THREADS%
echo TF_CPP_MIN_LOG_LEVEL=%TF_CPP_MIN_LOG_LEVEL%
echo.

echo Starting breast cancer classification project...
echo.

REM Run the project
python master_breast_cancer_classifier.py

echo.
echo Project completed!
pause 