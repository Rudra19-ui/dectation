#!/usr/bin/env python3
"""
Quick Status Check for Breast Cancer Classification System
"""

import subprocess
import sys

import requests


def check_backend():
    """Check if backend API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend API: RUNNING")
            print(f"   URL: http://localhost:8000")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print("❌ Backend API: NOT RESPONDING")
            return False
    except Exception as e:
        print(f"❌ Backend API: ERROR - {e}")
        return False


def check_frontend():
    """Check if frontend is running"""
    try:
        response = requests.get("http://localhost:8502", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend Web App: RUNNING")
            print(f"   URL: http://localhost:8502")
            return True
        else:
            print("❌ Frontend Web App: NOT RESPONDING")
            return False
    except Exception as e:
        print("❌ Frontend Web App: NOT RUNNING")
        return False


def check_ports():
    """Check which ports are in use"""
    print("\n🔍 Port Status:")
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        lines = result.stdout.split("\n")

        for line in lines:
            if ":8000" in line and "LISTENING" in line:
                print("   Port 8000: ✅ IN USE (Backend API)")
            elif ":8502" in line and "LISTENING" in line:
                print("   Port 8502: ✅ IN USE (Frontend Web App)")
    except:
        print("   Could not check port status")


def main():
    print("🏥 BREAST CANCER CLASSIFICATION SYSTEM STATUS")
    print("=" * 50)

    # Check backend
    backend_ok = check_backend()

    # Check frontend
    frontend_ok = check_frontend()

    # Check ports
    check_ports()

    print("\n" + "=" * 50)
    if backend_ok and frontend_ok:
        print("🎉 SYSTEM FULLY OPERATIONAL!")
        print("\n🚀 Ready to use:")
        print("   Web App: http://localhost:8502")
        print("   API Docs: http://localhost:8000/docs")
        print("   Health Check: http://localhost:8000/health")
    elif backend_ok:
        print("⚠️  Backend running, but frontend not detected")
        print("   Backend API: http://localhost:8000")
    else:
        print("❌ System not fully operational")
        print("   Try running: python backend_api.py")


if __name__ == "__main__":
    main()
