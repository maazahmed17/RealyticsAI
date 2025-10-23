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
        self.model_dir = Path("/home/maaz/RealyticsAI/data/models")
        
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
            # Create DataFrame from input
            df = pd.DataFrame([property_data])
            
            # Normalize column names
            df.columns = df.columns.str.lower()
            
            # Map column names if needed
            column_mapping = {
                'totalsqft': 'total_sqft',
                'size': 'bhk'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Extract BHK from size if needed
            if 'size' in df.columns and 'bhk' not in df.columns:
                df['bhk'] = df['size'].str.extract('(\d+)').astype(float)
            
            # Load reference data for statistical feature calculation
            data_path = Path("/home/maaz/RealyticsAI/data/raw/bengaluru_house_prices.csv")
            if data_path.exists():
                # Load a small sample of reference data for statistics
                import warnings
                warnings.filterwarnings('ignore')
                ref_data = pd.read_csv(data_path, nrows=1000)  # Load first 1000 rows
                ref_data.columns = ref_data.columns.str.lower()
                
                # Append the input to reference data
                combined_df = pd.concat([ref_data, df], ignore_index=True)
            else:
                combined_df = df
            
            # Apply feature engineering (NO DATA LEAKAGE)
            df_engineered = self.feature_engineer.transform(
                combined_df,
                use_polynomial=False,
                use_interactions=True,
                use_binning=True,
                use_statistical=True  # Enable for proper features
            )
            
            # Get only the last row (our input)
            df_engineered = df_engineered.tail(1)
            
            # Select only numeric features
            feature_cols = [col for col in df_engineered.columns if col != 'price']
            numeric_features = df_engineered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            X = df_engineered[numeric_features]
            
            # Handle missing values
            X = X.fillna(X.median() if len(X) > 1 else 0)
            
            # Align with training features
            if self.feature_columns:
                # Use only common features
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
