"""
EcoTransparency Platform - Main Application Entry Point
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.config.settings import Settings
from src.utils.logger import setup_logger

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logger()
    
    # Load settings
    settings = Settings()
    
    logger.info("Starting EcoTransparency Platform")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Database URL: {settings.database_url}")
    
    # Import and run the FastAPI app
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )

if __name__ == "__main__":
    main()
