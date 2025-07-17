"""
Configuration settings for the EcoTransparency Platform
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
    
    # Application settings
    app_name: str = "EcoTransparency Platform"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database settings
    database_url: str = Field(default="postgresql://localhost/ecotransparency", env="DATABASE_URL")
    mongo_url: str = Field(default="mongodb://localhost:27017/ecotransparency", env="MONGO_URL")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    google_vision_api_key: Optional[str] = Field(default=None, env="GOOGLE_VISION_API_KEY")
    
    # ML Model settings
    model_cache_dir: str = Field(default="models/cache", env="MODEL_CACHE_DIR")
    sustainability_model_path: str = Field(default="models/sustainability_index.pkl", env="SUSTAINABILITY_MODEL_PATH")
    greenwashing_model_path: str = Field(default="models/greenwashing_detector.pkl", env="GREENWASHING_MODEL_PATH")
    
    # Data processing settings
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    data_cache_ttl: int = Field(default=3600, env="DATA_CACHE_TTL")  # 1 hour
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")

# Global settings instance
settings = Settings()
