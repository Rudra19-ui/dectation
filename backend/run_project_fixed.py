#!/usr/bin/env python3
"""
Fixed Breast Cancer Classification Project Startup Script
Better error handling and process management
"""

import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import requests


def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("🏥 BREAST CANCER CLASSIFICATION PROJECT")
    print("=" * 80)
    print("🎯 Complete AI-powered medical imaging analysis system")
    print("📊 Model Accuracy: 92.74%")
    print("🚀 Multiple interfaces: API, Web App, CLI")
    print("=" * 80)


def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "streamlit",
        "PIL",
        "pandas",
        "numpy",
        "python-multipart",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "python-multipart":
                import multipart
            elif package == "PIL":
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")

    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")

        # Install missing packages
        for package in missing_packages:
            install_name = (
                "python-multipart" if package == "python-multipart" else package
            )
            install_name = "pillow" if package == "PIL" else install_name
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_name],
                    check=True,
                    capture_output=True,
                )
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False

    print("✅ All dependencies are installed!")
    return True


def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")

    model_files = ["best_improved_model.pt", "quick_enhanced_model.pt"]

    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"❌ {model_file} - NOT FOUND")

    return len(found_models) > 0


def test_model_loading():
    """Test if models can be loaded successfully"""
    print("\n🔍 Testing model loading...")

    try:
        # Test Streamlit app model loading
        from image_upload_app import BreastCancerPredictor

        predictor = BreastCancerPredictor()
        if predictor.model is not None:
            print("✅ Streamlit model loads successfully")
        else:
            print("❌ Streamlit model failed to load")
            return False
    except Exception as e:
        print(f"❌ Streamlit model loading error: {e}")
        return False

    try:
        # Test backend API model loading
        from backend_api import BreastCancerPredictor as BackendPredictor

        backend_predictor = BackendPredictor()
        if backend_predictor.model is not None:
            print("✅ Backend model loads successfully")
        else:
            print("❌ Backend model failed to load")
            return False
    except Exception as e:
        print(f"❌ Backend model loading error: {e}")
        return False

    return True


def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting Backend API Server...")

    try:
        # Start backend process
        backend_process = subprocess.Popen(
            [sys.executable, "backend_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start with proper timeout
        for i in range(30):  # 30 second timeout
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend API is running on http://localhost:8000")
                    print("📚 API Documentation: http://localhost:8000/docs")
                    return backend_process
            except requests.exceptions.RequestException:
                continue

        print("❌ Backend API failed to start within timeout")
        backend_process.terminate()
        return None

    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None


def start_streamlit():
    """Start the Streamlit frontend"""
    print("\n🌐 Starting Streamlit Web App...")

    try:
        # Start Streamlit process
        streamlit_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "image_upload_app.py",
                "--server.port",
                "8502",
                "--server.headless",
                "true",
                "--server.runOnSave",
                "false",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it time to start
        time.sleep(8)

        print("✅ Streamlit Web App is running on http://localhost:8502")
        return streamlit_process

    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None


def open_browsers():
    """Open browsers to the applications"""
    print("\n🌐 Opening applications in browser...")

    urls = [
        ("Streamlit App", "http://localhost:8502"),
        ("API Documentation", "http://localhost:8000/docs"),
    ]

    for name, url in urls:
        try:
            webbrowser.open(url)
            print(f"✅ Opened: {name} - {url}")
            time.sleep(1)  # Small delay between opens
        except Exception as e:
            print(f"❌ Could not open {name}: {e}")


def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "=" * 80)
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 80)
    print("🎯 You can now use the system in multiple ways:")
    print()
    print("1. 🌐 Web Interface (Streamlit):")
    print("   - URL: http://localhost:8502")
    print("   - Features: Drag & drop upload, real-time predictions")
    print()
    print("2. 🔌 API Endpoints:")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Model Info: http://localhost:8000/model-info")
    print("   - Predict: POST http://localhost:8000/predict")
    print("   - Documentation: http://localhost:8000/docs")
    print()
    print("3. 💻 Command Line:")
    print("   - Single image: python predict_uploaded_image.py image.png")
    print("   - Training: python improved_training.py")
    print()
    print("⚠️  IMPORTANT: This is an educational tool only!")
    print("   Always consult healthcare professionals for medical decisions.")
    print("=" * 80)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down services...")
    sys.exit(0)


def main():
    """Main startup function"""
    signal.signal(signal.SIGINT, signal_handler)

    print_banner()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return

    # Check model files
    if not check_model_files():
        print("\n⚠️ No model files found. Some features may not work.")

    # Test model loading
    if not test_model_loading():
        print("\n⚠️ Model loading issues detected. Continuing anyway...")

    # Start services
    backend_process = None
    streamlit_process = None

    try:
        # Start backend
        backend_process = start_backend()

        # Start frontend
        streamlit_process = start_streamlit()

        if not streamlit_process:
            print("\n❌ Failed to start Streamlit. Backend may still be available.")

        # Open browsers
        if backend_process or streamlit_process:
            time.sleep(2)  # Give services time to fully start
            open_browsers()
            show_usage_instructions()

            print("\n🎉 Project is now running!")
            print("Press Ctrl+C to stop all services...")

            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("\n❌ Failed to start any services.")

    finally:
        # Cleanup
        print("\n\n🛑 Shutting down services...")

        if backend_process:
            backend_process.terminate()
            print("✅ Backend API stopped")

        if streamlit_process:
            streamlit_process.terminate()
            print("✅ Streamlit stopped")

        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
