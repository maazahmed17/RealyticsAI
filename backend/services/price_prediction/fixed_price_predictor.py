"""
Fixed Price Prediction Service (NO DATA LEAKAGE)
=================================================
Production-ready price predictor using the newly trained fixed model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Add to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
from models.feature_engineering_advanced import AdvancedFeatureEngineer


class FixedPricePredictionService:
    """
    Production-ready price prediction service using the fixed model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_dir = Path("/home/maaz/RealyticsAI_Dev/data/models")
        
        # Load the fixed model
        self._load_fixed_model()
    
    def _load_fixed_model(self):
        """Load the latest fixed model"""
        try:
            # Find latest fixed model
            fixed_models = list(self.model_dir.glob("xgboost_fixed_*.pkl"))
            
            if not fixed_models:
                logger.error("No fixed model found! Please train the model first.")
                return
            
            # Load latest
            latest_model_path = max(fixed_models, key=lambda p: p.stat().st_ctime)
            timestamp = latest_model_path.name.replace("xgboost_fixed_", "").replace(".pkl", "")
            
            # Load model
            self.model = joblib.load(latest_model_path)
            logger.info(f"✅ Loaded fixed model: {latest_model_path.name}")
            
            # Load scaler
            scaler_path = self.model_dir / f"scaler_{timestamp}.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler")
            
            # Load feature columns
            features_path = self.model_dir / f"feature_columns_{timestamp}.pkl"
            if features_path.exists():
                self.feature_columns = joblib.load(features_path)
                logger.info(f"✅ Loaded {len(self.feature_columns)} feature columns")
            
        except Exception as e:
            logger.error(f"Error loading fixed model: {e}")
    
    def predict(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict property price
        
        Parameters:
        -----------
        property_data : dict
            Dictionary with keys: location, bhk, bath, balcony, total_sqft
            
        Returns:
        --------
        dict : Prediction result with price and confidence
        """
        if not self.model:
            return {
                "success": False,
                "error": "Model not loaded",
                "price": None
            }
        
        try:
            # Extract basic features
            bhk = property_data.get('bhk', property_data.get('bedrooms', 3))
            bath = property_data.get('bath', property_data.get('bathrooms', 2))
            balcony = property_data.get('balcony', property_data.get('balconies', 1))
            sqft = property_data.get('total_sqft', property_data.get('area', property_data.get('sqft', None)))
            
            # Handle missing sqft - estimate based on BHK
            if not sqft:
                sqft_estimates = {1: 650, 2: 1100, 3: 1650, 4: 2200, 5: 2800}
                sqft = sqft_estimates.get(bhk, 1650)
            
            # Load reference data for location encoding
            data_path = Path("/home/maaz/RealyticsAI_Dev/data/bengaluru_house_prices.csv")
            if data_path.exists():
                ref_data = pd.read_csv(data_path)
                # Map location to frequency and price encoding
                location_counts = ref_data['Location'].value_counts()
                location_prices = ref_data.groupby('Location')['Price'].mean()
                global_mean = ref_data['Price'].mean()
                
                location = property_data.get('location', 'Bangalore')
                loc_freq = location_counts.get(location, 10)  # Default frequency
                loc_price = location_prices.get(location, global_mean)  # Default price encoding
            else:
                loc_freq = 10
                loc_price = 200  # Default price encoding
            
            # Create feature dictionary matching new training format
            feature_dict = {
                'BHK': bhk,
                'TotalSqft': sqft,
                'Bath': bath,
                'Balcony': balcony,
                'PropertyAgeYears': property_data.get('property_age', 5),
                'FloorNumber': property_data.get('floor_number', 2),
                'TotalFloors': property_data.get('total_floors', 4),
                'Parking': property_data.get('parking', 1),
                'LocationFrequency': loc_freq,
                'LocationPriceEncoding': loc_price,
                'SqftPerBHK': sqft / bhk,
                'BathPerBHK': bath / bhk,
                'TotalRooms': bhk + bath
            }
            
            # Create DataFrame
            X = pd.DataFrame([feature_dict])
            
            # Align with training features
            if self.feature_columns:
                # Use only the features that exist in both
                common_features = [f for f in self.feature_columns if f in X.columns]
                X = X[common_features]
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Predict
            predicted_price = self.model.predict(X_scaled)[0]
            
            # Calculate confidence based on feature completeness
            feature_completeness = len(X.columns) / len(self.feature_columns) if self.feature_columns else 1.0
            confidence = min(0.95, feature_completeness * 0.9)
            
            return {
                "success": True,
                "price": float(predicted_price),
                "price_formatted": f"₹{predicted_price:.2f} Lakhs",
                "confidence": confidence,
                "features_used": len(X.columns),
                "model": "xgboost_fixed (No Data Leakage)"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "success": False,
                "error": str(e),
                "price": None
            }
    
    def predict_batch(self, properties: list) -> list:
        """Predict prices for multiple properties"""
        results = []
        for prop in properties:
            result = self.predict(prop)
            results.append(result)
        return results


# Singleton instance
_service_instance = None

def get_price_predictor():
    """Get or create the price predictor service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = FixedPricePredictionService()
    return _service_instance
