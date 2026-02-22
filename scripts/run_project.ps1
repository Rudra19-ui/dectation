# PowerShell script to run breast cancer classification project
# Sets environment variables to avoid multiprocessing issues

Write-Host "========================================" -ForegroundColor Green
Write-Host "Breast Cancer Classification Project" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Set environment variables to avoid multiprocessing issues
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONPATH = "."

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "OMP_NUM_THREADS: $env:OMP_NUM_THREADS" -ForegroundColor Cyan
Write-Host "MKL_NUM_THREADS: $env:MKL_NUM_THREADS" -ForegroundColor Cyan
Write-Host "TF_CPP_MIN_LOG_LEVEL: $env:TF_CPP_MIN_LOG_LEVEL" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting breast cancer classification project..." -ForegroundColor Yellow
Write-Host ""

# Run the project
python master_breast_cancer_classifier.py

Write-Host ""
Write-Host "Project completed!" -ForegroundColor Green
Read-Host "Press Enter to exit" 