#!/usr/bin/env python3
"""
Startup script for EcoTransparency Platform
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")

def start_streamlit_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Streamlit dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/streamlit_app.py", 
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    try:
        subprocess.run([sys.executable, "test_platform.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")

def run_examples():
    """Run example scripts"""
    print("📖 Running examples...")
    try:
        subprocess.run([sys.executable, "examples/usage_examples.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Examples failed: {e}")

def setup_environment():
    """Setup the environment"""
    print("🔧 Setting up environment...")
    try:
        subprocess.run([sys.executable, "setup.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")

def install_dependencies():
    """Install dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")

def show_status():
    """Show platform status"""
    print("📊 EcoTransparency Platform Status")
    print("=" * 40)
    
    # Check if files exist
    required_files = [
        "main.py",
        "requirements.txt",
        "src/api/main.py",
        "frontend/streamlit_app.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    if check_dependencies():
        print("✅ All dependencies installed")
    else:
        print("❌ Some dependencies missing")
    
    # Check environment file
    if os.path.exists(".env"):
        print("✅ .env file exists")
    else:
        print("❌ .env file missing")
    
    # Check directories
    required_dirs = ["src", "frontend", "models", "logs"]
    print("\n📁 Directories:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="EcoTransparency Platform Startup Script")
    parser.add_argument("command", nargs="?", default="help", 
                       choices=["api", "dashboard", "both", "test", "examples", "setup", "install", "status", "help"],
                       help="Command to run")
    
    args = parser.parse_args()
    
    print("🌱 EcoTransparency Platform")
    print("=" * 30)
    
    if args.command == "help":
        print("Available commands:")
        print("  api        - Start FastAPI server")
        print("  dashboard  - Start Streamlit dashboard")
        print("  both       - Start both API and dashboard")
        print("  test       - Run tests")
        print("  examples   - Run examples")
        print("  setup      - Setup environment")
        print("  install    - Install dependencies")
        print("  status     - Show platform status")
        print("  help       - Show this help")
        
        print("\nQuick start:")
        print("1. python start.py install")
        print("2. python start.py setup")
        print("3. python start.py dashboard")
        
    elif args.command == "api":
        if not check_dependencies():
            return
        start_api_server()
        
    elif args.command == "dashboard":
        if not check_dependencies():
            return
        start_streamlit_dashboard()
        
    elif args.command == "both":
        if not check_dependencies():
            return
        print("🚀 Starting both API server and dashboard...")
        print("Note: This will start the API server first, then the dashboard")
        print("Use separate terminals for better control")
        start_api_server()
        
    elif args.command == "test":
        run_tests()
        
    elif args.command == "examples":
        run_examples()
        
    elif args.command == "setup":
        setup_environment()
        
    elif args.command == "install":
        install_dependencies()
        
    elif args.command == "status":
        show_status()

if __name__ == "__main__":
    main()
