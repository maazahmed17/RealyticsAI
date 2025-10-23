"""
RealyticsAI Configuration Settings
==================================
Centralized configuration for the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "bengaluru_house_prices.csv"
MODEL_DIR = BASE_DIR / "data" / "models"

# Model Configuration
MODEL_ACCURACY = 0.9957  # R² score
MODEL_RMSE = 3.50  # in Lakhs

# Application Settings
APP_NAME = "RealyticsAI"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Intelligent Real Estate Assistant for Bengaluru"
