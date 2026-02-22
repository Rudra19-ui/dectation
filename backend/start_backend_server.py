#!/usr/bin/env python3
"""
Backend Server Startup Script
Starts the FastAPI backend server for breast cancer classification
"""

import os
import subprocess
import sys
import time

import requests


def start_backend_server():
    """Start the backend server"""
    print("🚀 Starting Breast Cancer Classification Backend Server...")

    try:
        # Start the server
        server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend_api:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )

        print("⏳ Waiting for server to start...")

        # Wait for server to be ready
        server_ready = False
        for i in range(30):  # 30 second timeout
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    print("✅ Backend server is running!")
                    print(f"🌐 Server URL: http://localhost:8000")
                    print(f"📚 API Documentation: http://localhost:8000/docs")
                    print(f"🏥 Health Check: http://localhost:8000/health")
                    break
            except requests.exceptions.RequestException:
                continue

        if not server_ready:
            print("❌ Server failed to start within timeout")
            server_process.terminate()
            return False

        try:
            # Test the endpoints
            print("\n🧪 Testing endpoints...")

            # Test health endpoint
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("✅ Health endpoint working")

            # Test model info endpoint
            response = requests.get("http://localhost:8000/model-info")
            if response.status_code == 200:
                model_info = response.json()
                print("✅ Model info endpoint working")
                print(f"   Model: {model_info.get('model_type', 'N/A')}")
                print(f"   Classes: {model_info.get('classes', 'N/A')}")

            print("\n🎉 Server is fully functional!")
            print("\n📋 Available endpoints:")
            print("   GET  /           - API info")
            print("   GET  /health     - Health check")
            print("   GET  /model-info - Model information")
            print("   POST /predict    - Single image prediction")
            print("   POST /predict-batch - Multiple image prediction")
            print("   GET  /docs       - Interactive API documentation")

            # Keep server running
            print("\n🔄 Server running... Press Ctrl+C to stop")
            server_process.wait()

        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            server_process.terminate()

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True


if __name__ == "__main__":
    start_backend_server()
