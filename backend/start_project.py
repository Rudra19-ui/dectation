#!/usr/bin/env python3
"""
Breast Cancer Classification Project - Startup Script
Launches both backend API and frontend applications
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


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
        "requests",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")

    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False

    print("✅ All dependencies are installed!")
    return True


def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")

    model_files = ["best_improved_model.pt", "quick_enhanced_model.pt"]

    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_file} - NOT FOUND")

    return True


def start_backend():
    """Start the FastAPI backend server"""
    print("\n🚀 Starting Backend API Server...")

    try:
        # Start backend in background
        backend_process = subprocess.Popen(
            [sys.executable, "backend_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Poll for server readiness (up to 30 seconds)
        import requests

        max_wait_seconds = 30
        poll_interval_seconds = 1
        waited = 0
        while waited < max_wait_seconds:
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend API is running on http://localhost:8000")
                    print("📚 API Documentation: http://localhost:8000/docs")
                    return backend_process
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval_seconds)
            waited += poll_interval_seconds

        # If not ready, surface stderr to aid debugging
        try:
            stderr_output = (
                backend_process.stderr.read().decode(errors="ignore")
                if backend_process.stderr
                else ""
            )
        except Exception:
            stderr_output = ""
        print("❌ Backend API did not respond within timeout")
        if stderr_output:
            print("--- Backend stderr ---")
            print(stderr_output)
            print("----------------------")
        return None

    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None


def start_frontend():
    """Start the Streamlit frontend"""
    print("\n🌐 Starting Frontend Web App...")

    try:
        # Start Streamlit in background
        frontend_process = subprocess.Popen(
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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(5)

        print("✅ Frontend Web App is running on http://localhost:8502")
        return frontend_process

    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None


def open_browsers():
    """Open browsers to the applications"""
    print("\n🌐 Opening applications in browser...")

    urls = [
        "http://localhost:8000/docs",  # API Documentation
        "http://localhost:8502",  # Streamlit App
    ]

    for url in urls:
        try:
            webbrowser.open(url)
            print(f"✅ Opened: {url}")
        except Exception as e:
            print(f"❌ Could not open {url}: {e}")


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
    print("2. 🌐 HTML Frontend:")
    print("   - File: frontend.html (open in browser)")
    print("   - Features: Modern UI, connects to API")
    print()
    print("3. 🔌 API Endpoints:")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Model Info: http://localhost:8000/model-info")
    print("   - Predict: POST http://localhost:8000/predict")
    print("   - Documentation: http://localhost:8000/docs")
    print()
    print("4. 💻 Command Line:")
    print("   - Single image: python predict_uploaded_image.py image.png")
    print("   - Training: python improved_training.py")
    print()
    print("⚠️  IMPORTANT: This is an educational tool only!")
    print("   Always consult healthcare professionals for medical decisions.")
    print("=" * 80)


def get_project_about_text() -> str:
    return (
        "Breast Cancer Classification Project\n"
        "\n"
        "What this project is:\n"
        "- End-to-end system: FastAPI backend, Streamlit web app, CLI tools.\n"
        "- Primary model: best_improved_model.pt (ResNet50-based; Normal/Benign/Malignant).\n"
        "- Explainability (Grad-CAM), evaluation utilities, PDF reports.\n"
        "\n"
        "What this project is NOT:\n"
        "- Not a medical device or substitute for clinical diagnosis.\n"
        "- Not a native desktop GUI; web UI runs locally in a browser.\n"
        "\n"
        "Key entry points:\n"
        "- start_project.py (launcher), start_backend_server.py (API only), image_upload_app.py (UI).\n"
        "- CLI prediction: predict_uploaded_image.py <image>.\n"
        "\n"
        "Packaging notes:\n"
        "- PyInstaller can package start_project.py, but it spawns uvicorn/streamlit at runtime.\n"
        "- For fully offline distribution, additional bundling work is required.\n"
        "\n"
        "Disclaimers:\n"
        "- Educational use only; no warranties; consult professionals for medical decisions.\n"
    )


def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(
        description="Breast Cancer Classification Project Launcher"
    )
    parser.add_argument(
        "--about",
        action="store_true",
        help="Print detailed project information and exit",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version information and exit"
    )
    args = parser.parse_args()

    if args.version:
        print("Breast Cancer Classification Project - Version 1.0.0")
        return

    if args.about:
        print(get_project_about_text())
        return

    print_banner()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return

    # Check model files
    check_model_files()

    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend. Exiting.")
        return

    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend. Backend is still running.")

    # Open browsers
    open_browsers()

    # Show instructions
    show_usage_instructions()

    print("\n🎉 Project is now running!")
    print("Press Ctrl+C to stop all services...")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")

        if backend_process:
            backend_process.terminate()
            print("✅ Backend API stopped")

        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")

        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
