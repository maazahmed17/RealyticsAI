"""
Configuration Management for RealyticsAI
=========================================
Centralized configuration for all system settings and API keys.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """System-wide configuration settings"""
    
    # API Keys
    GEMINI_API_KEY: str = Field(
        default=None,
        env="GEMINI_API_KEY"
    )
    
    # Gemini Configuration
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    GEMINI_TEMPERATURE: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    GEMINI_MAX_TOKENS: int = Field(default=2048, env="GEMINI_MAX_TOKENS")
    
    # System Configuration
    APP_NAME: str = "RealyticsAI"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE")
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    BENGALURU_DATA_PATH: str = "/mnt/c/Users/Ahmed/Downloads/bengaluru_house_prices.csv"
    
    # Model Paths (will be updated dynamically)
    PRICE_MODEL_PATH: Optional[Path] = None
    FEATURE_LIST_PATH: Optional[Path] = None
    SCALER_PATH: Optional[Path] = None
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns",
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT: str = Field(
        default="realyticsai_complete",
        env="MLFLOW_EXPERIMENT"
    )
    
    # Feature Flags
    ENABLE_PRICE_PREDICTION: bool = True
    ENABLE_PROPERTY_RECOMMENDATION: bool = True  # Now enabled with recommendation engine
    ENABLE_NEGOTIATION_AGENT: bool = False  # Will be enabled when implemented
    ENABLE_INTELLIGENT_ROUTING: bool = True  # Enable smart query routing
    
    # Chatbot Configuration
    CHATBOT_CONTEXT_LENGTH: int = Field(default=10, env="CHATBOT_CONTEXT_LENGTH")
    CHATBOT_RESPONSE_FORMAT: str = Field(default="detailed", env="CHATBOT_RESPONSE_FORMAT")
    
    # Recommendation System Configuration
    RECOMMENDATION_MODEL_TYPE: str = Field(default="tfidf", env="RECOMMENDATION_MODEL_TYPE")
    RECOMMENDATION_TOP_K_DEFAULT: int = Field(default=10, env="RECOMMENDATION_TOP_K_DEFAULT")
    RECOMMENDATION_DATA_PATH: str = Field(
        default="backend/services/recommendation_service/data/dataproperties.csv", 
        env="RECOMMENDATION_DATA_PATH"
    )
    ENABLE_BERT_RECOMMENDATIONS: bool = Field(default=False, env="ENABLE_BERT_RECOMMENDATIONS")
    ENABLE_HYBRID_RECOMMENDATIONS: bool = Field(default=False, env="ENABLE_HYBRID_RECOMMENDATIONS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_latest_model_paths(self):
        """Get the latest model paths from the models directory"""
        if not self.MODELS_DIR.exists():
            return False
            
        # Find latest model files
        model_files = list(self.MODELS_DIR.glob("enhanced_model_*.pkl"))
        feature_files = list(self.MODELS_DIR.glob("features_*.txt"))
        scaler_files = list(self.MODELS_DIR.glob("scaler_*.pkl"))
        
        if model_files:
            self.PRICE_MODEL_PATH = max(model_files, key=os.path.getctime)
        if feature_files:
            self.FEATURE_LIST_PATH = max(feature_files, key=os.path.getctime)
        if scaler_files:
            self.SCALER_PATH = max(scaler_files, key=os.path.getctime)
            
        return bool(self.PRICE_MODEL_PATH)
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present"""
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your-api-key-here":
            return False
        return True
    
    @classmethod
    def validate_api_keys_static(cls) -> bool:
        """Static method for API key validation"""
        settings_instance = cls()
        return settings_instance.validate_api_keys()


# Create global settings instance
settings = Settings()

# Auto-detect latest model paths
settings.get_latest_model_paths()


# Export settings
__all__ = ["settings", "Settings"]
