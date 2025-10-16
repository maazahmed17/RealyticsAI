"""
RealyticsAI - Application Settings Configuration
Centralized configuration management for the platform
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Configuration
    app_name: str = "RealyticsAI"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Configuration
    database_url: Optional[str] = None
    db_echo: bool = False
    
    # Redis Configuration (for caching)
    redis_url: str = "redis://localhost:6379/0"
    
    # ML Model Configuration
    model_path: str = "../data/models"
    data_path: str = "../data"
    
    # Price Prediction Service
    price_model_name: str = "local_prices_predictor"
    default_data_file: str = "data/bengaluru_house_prices.csv"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS Settings
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080"
    ]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Flags
    enable_price_prediction: bool = True
    enable_property_recommendation: bool = False
    enable_negotiation_agent: bool = False
    enable_chatbot: bool = False
    
    # External APIs (for future integrations)
    openai_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None
    
    # File Upload Settings
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".csv", ".xlsx", ".xls", ".zip"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Environment-specific settings
def get_database_url() -> str:
    """Get database URL based on environment"""
    settings = get_settings()
    
    if settings.environment == "production":
        return os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/realyticsai")
    elif settings.environment == "testing":
        return "sqlite:///./test_realyticsai.db"
    else:
        return "sqlite:///./realyticsai.db"


def get_redis_url() -> str:
    """Get Redis URL based on environment"""
    settings = get_settings()
    
    if settings.environment == "production":
        return os.getenv("REDIS_URL", settings.redis_url)
    else:
        return settings.redis_url
