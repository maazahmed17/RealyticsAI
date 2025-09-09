"""
RealyticsAI Configuration Settings
==================================
Centralized configuration for the application.
"""

from pathlib import Path

# API Keys
GEMINI_API_KEY = "AIzaSyBS5TJCebmoyy9QyE_R-OAaYYV9V2oM-A8"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = "/mnt/c/Users/Ahmed/Downloads/bengaluru_house_prices.csv"
MODEL_DIR = BASE_DIR / "data" / "models"

# Model Configuration
MODEL_ACCURACY = 0.9957  # R² score
MODEL_RMSE = 3.50  # in Lakhs

# Application Settings
APP_NAME = "RealyticsAI"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Intelligent Real Estate Assistant for Bengaluru"
