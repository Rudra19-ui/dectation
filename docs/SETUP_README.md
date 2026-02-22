# Breast Cancer Classification - Environment Setup

This guide will help you set up the environment for the breast cancer classification project on Windows.

## 🚀 Quick Setup

### Option 1: Python Script (Recommended)
```bash
python setup_environment.py
```

### Option 2: PowerShell Script
```powershell
.\setup_environment.ps1
```

### Option 3: Batch Script
```cmd
setup_environment.bat
```

## 📦 What Gets Installed

The setup scripts will install the following packages:

- **tensorflow** - Deep learning framework
- **opencv-python** - Computer vision library
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting library
- **numpy** - Numerical computing
- **pillow** - Image processing
- **seaborn** - Statistical visualization
- **pydicom** - DICOM medical image handling

## 🔧 Prerequisites

1. **Python 3.7+** installed and added to PATH
2. **Windows 10/11** (scripts are optimized for Windows)
3. **Internet connection** for downloading packages

## 📋 Step-by-Step Instructions

### Step 1: Download the Setup Scripts
Make sure you have these files in your project directory:
- `setup_environment.py` (Python script)
- `setup_environment.ps1` (PowerShell script)
- `setup_environment.bat` (Batch script)

### Step 2: Run the Setup
Choose one of the following methods:

#### Method A: Python Script (Cross-platform)
```bash
python setup_environment.py
```

#### Method B: PowerShell (Windows)
```powershell
.\setup_environment.ps1
```

#### Method C: Command Prompt (Windows)
```cmd
setup_environment.bat
```

### Step 3: Wait for Installation
The script will:
1. ✅ Create a virtual environment called `breast_cancer_env`
2. ✅ Activate the virtual environment
3. ✅ Upgrade pip to the latest version
4. ✅ Install all required packages
5. ✅ Test the installation

### Step 4: Activate Environment (Future Use)
After setup, activate the environment before running the project:

#### In Command Prompt:
```cmd
breast_cancer_env\Scripts\activate.bat
```

#### In PowerShell:
```powershell
.\breast_cancer_env\Scripts\Activate.ps1
```

### Step 5: Run the Project
Once the environment is activated, run the breast cancer classifier:
```bash
python simple_classifier.py
```

## 🛠️ Troubleshooting

### Issue: "Python not found"
**Solution**: Install Python from [python.org](https://python.org) and add it to PATH

### Issue: "Permission denied"
**Solution**: Run PowerShell as Administrator or use the Python script

### Issue: "Failed to create virtual environment"
**Solution**: 
1. Ensure Python is installed correctly
2. Try running: `python -m venv --help`
3. Check if you have write permissions in the directory

### Issue: "Package installation failed"
**Solution**:
1. Check your internet connection
2. Try updating pip: `python -m pip install --upgrade pip`
3. Install packages individually if needed

### Issue: "TensorFlow installation slow"
**Solution**: 
1. This is normal for TensorFlow
2. Consider using a faster internet connection
3. The installation may take 10-30 minutes

## 📁 Project Structure After Setup

```
E:\rudra\project\
├── breast_cancer_env\          # Virtual environment
│   ├── Scripts\
│   │   ├── python.exe
│   │   ├── pip.exe
│   │   └── activate.bat
│   └── Lib\site-packages\      # Installed packages
├── setup_environment.py        # Python setup script
├── setup_environment.ps1       # PowerShell setup script
├── setup_environment.bat       # Batch setup script
├── simple_classifier.py        # Main classifier script
├── split_dataset\              # Your dataset
└── SETUP_README.md            # This file
```

## 🎯 Expected Output

When the setup completes successfully, you should see:

```
============================================================
🎉 Setup Complete!
============================================================

✅ tensorflow: 2.x.x
✅ opencv: 4.x.x
✅ scikit-learn: 1.x.x
✅ matplotlib: 3.x.x
✅ numpy: 1.x.x

To activate the environment:
  breast_cancer_env\Scripts\activate.bat

To run the breast cancer classifier:
  python simple_classifier.py
```

## 🔍 Verification

To verify the installation, run:
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

## 🚀 Next Steps

After successful setup:

1. **Activate the environment**:
   ```bash
   breast_cancer_env\Scripts\activate.bat
   ```

2. **Run the classifier**:
   ```bash
   python simple_classifier.py
   ```

3. **Check your dataset**:
   Ensure `split_dataset` directory exists with train/val/test subdirectories

## 📞 Support

If you encounter issues:

1. **Check Python installation**: `python --version`
2. **Check pip installation**: `pip --version`
3. **Try the Python script** (most reliable)
4. **Run as Administrator** if permission issues occur

## 🎉 Success!

Once setup is complete, you're ready to run the breast cancer classification project with all the required dependencies installed in a clean virtual environment. 