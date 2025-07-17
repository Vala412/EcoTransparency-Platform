#!/usr/bin/env python3
"""
Setup script for EcoTransparency Platform
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                            "Installing requirements")
        if success:
            print("✅ All dependencies installed successfully")
        else:
            print("❌ Failed to install some dependencies")
            return False
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Install spaCy model
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", 
                "Installing spaCy English model")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models",
        "logs",
        "temp",
        "exports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def create_env_file():
    """Create a sample .env file"""
    env_content = """# Database Configuration
DATABASE_URL=postgresql://eco_user:eco_password@localhost/ecotransparency
MONGODB_URL=mongodb://localhost:27017/ecotransparency

# API Configuration
SECRET_KEY=your-secret-key-here-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External APIs (optional)
WEATHER_API_KEY=your-weather-api-key
SUPPLY_CHAIN_API_KEY=your-supply-chain-api-key

# Environment
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# Model Paths
SUSTAINABILITY_MODEL_PATH=models/sustainability_model.pkl
GREENWASHING_MODEL_PATH=models/greenwashing_model.pkl
FORECASTING_MODEL_PATH=models/forecasting_model.pkl
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("📄 .env file already exists")

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
        else:
            print("❌ Docker is not available")
            return False
    except:
        print("❌ Docker is not installed")
        return False

def setup_database():
    """Set up database using Docker"""
    if not check_docker():
        print("⚠️  Docker not available. Please install PostgreSQL and MongoDB manually.")
        return False
    
    print("🐳 Setting up databases with Docker...")
    
    # PostgreSQL
    postgres_cmd = """docker run -d --name postgres-eco \
        -e POSTGRES_DB=ecotransparency \
        -e POSTGRES_USER=eco_user \
        -e POSTGRES_PASSWORD=eco_password \
        -p 5432:5432 postgres:13"""
    
    success = run_command(postgres_cmd, "Setting up PostgreSQL")
    
    # MongoDB
    mongo_cmd = "docker run -d --name mongodb-eco -p 27017:27017 mongo:4.4"
    success = run_command(mongo_cmd, "Setting up MongoDB")
    
    return success

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    # Test imports
    test_imports = [
        "fastapi",
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "spacy"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {module}")
            return False
    
    print("✅ All modules imported successfully")
    return True

def main():
    """Main setup function"""
    print("🌱 EcoTransparency Platform Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update the .env file with your configuration")
    print("2. Set up databases (PostgreSQL and MongoDB)")
    print("3. Run the API server: python main.py")
    print("4. Run the dashboard: streamlit run frontend/streamlit_app.py")
    print("\nFor database setup with Docker:")
    print("- Run: python setup.py --database")
    print("\nFor help: python setup.py --help")

def show_help():
    """Show help information"""
    print("EcoTransparency Platform Setup Script")
    print("\nUsage:")
    print("  python setup.py           # Full setup")
    print("  python setup.py --database # Setup databases only")
    print("  python setup.py --help    # Show this help")
    print("\nOptions:")
    print("  --database    Setup PostgreSQL and MongoDB with Docker")
    print("  --help        Show this help message")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            show_help()
        elif sys.argv[1] == "--database":
            setup_database()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            show_help()
    else:
        main()
