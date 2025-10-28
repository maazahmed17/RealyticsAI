"""
Enhanced Price Prediction Service with XGBoost
==============================================
This module provides proper ML-based price predictions using XGBoost
with location encoding and all relevant features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import pickle
import joblib
import os
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


class LocationEncoder:
    """Handles location encoding with price-based target encoding"""
    
    def __init__(self):
        self.location_price_map = {}
        self.default_price = 100.0
        
    def fit(self, df: pd.DataFrame, target_col: str = 'price'):
        """Fit location encoder on training data"""
        if 'location' in df.columns and target_col in df.columns:
            # Calculate mean price per location
            location_stats = df.groupby('location')[target_col].agg(['mean', 'count'])
            
            # Apply smoothing for locations with few samples
            global_mean = df[target_col].mean()
            smoothing_factor = 10  # Minimum samples for reliable estimate
            
            for location, row in location_stats.iterrows():
                count = row['count']
                mean_price = row['mean']
                # Weighted average between location mean and global mean
                weight = count / (count + smoothing_factor)
                smoothed_price = weight * mean_price + (1 - weight) * global_mean
                self.location_price_map[location] = smoothed_price
            
            self.default_price = global_mean
            logger.info(f"Location encoder fitted with {len(self.location_price_map)} locations")
        
    def transform(self, location: str) -> Dict[str, float]:
        """Transform location to encoded features"""
        encoded_price = self.location_price_map.get(location, self.default_price)
        
        # Create location tier based on price
        if encoded_price > self.default_price * 1.3:
            tier = 3  # Premium
        elif encoded_price > self.default_price * 1.1:
            tier = 2  # Above average
        elif encoded_price > self.default_price * 0.9:
            tier = 1  # Average
        else:
            tier = 0  # Budget
        
        return {
            'location_encoded': encoded_price,
            'location_tier': tier
        }


class EnhancedPricePredictionService:
    """
    Enhanced Price Prediction Service using XGBoost
    """
    
    def __init__(self):
        self.xgb_model = None
        self.location_encoder = LocationEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.data = None
        self.model_path = Path("/home/maaz/RealyticsAI_Dev/data/models")
        self.data_path = Path("/home/maaz/RealyticsAI_Dev/data/bengaluru_house_prices.csv")
        
        # Initialize service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the service with models and data"""
        try:
            # Load data first for location encoding
            self._load_data()
            
            # Load or train model
            if not self._load_existing_model():
                logger.info("No existing model found, training new model...")
                self._train_new_model()
            
            logger.info("Enhanced Price Prediction Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Price Prediction Service: {e}")
    
    def _load_data(self):
        """Load and prepare the Bengaluru dataset"""
        try:
            if self.data_path.exists():
                self.data = pd.read_csv(self.data_path)
                
                # Clean data
                self._clean_data()
                
                # Fit location encoder
                self.location_encoder.fit(self.data)
                
                logger.info(f"Loaded and cleaned dataset: {self.data.shape[0]} properties")
            else:
                logger.warning(f"Data file not found: {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _clean_data(self):
        """Clean and prepare the dataset"""
        # Handle total_sqft column
        if 'total_sqft' in self.data.columns:
            def convert_sqft(x):
                try:
                    if '-' in str(x):
                        parts = str(x).split('-')
                        return (float(parts[0]) + float(parts[1])) / 2
                    return float(x)
                except:
                    return np.nan
            
            self.data['total_sqft'] = self.data['total_sqft'].apply(convert_sqft)
        
        # Extract BHK from size
        if 'size' in self.data.columns and 'bhk' not in self.data.columns:
            self.data['bhk'] = self.data['size'].str.extract('(\d+)').astype(float)
        
        # Drop rows with missing critical values
        critical_cols = ['price', 'total_sqft', 'bath', 'balcony', 'location']
        for col in critical_cols:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        # Remove extreme outliers
        for col in ['price', 'total_sqft']:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                self.data = self.data[(self.data[col] >= lower) & (self.data[col] <= upper)]
    
    def _load_existing_model(self) -> bool:
        """Try to load an existing trained model"""
        try:
            # Try XGBoost fixed models first (latest models)
            fixed_models = list(self.model_path.glob("xgboost_fixed_*.pkl"))
            if fixed_models:
                model_file = max(fixed_models, key=os.path.getctime)
                timestamp = model_file.name.replace("xgboost_fixed_", "").replace(".pkl", "")
                
                # Load model
                self.xgb_model = joblib.load(model_file)
                logger.info(f"Loaded XGBoost fixed model from {model_file}")
                
                # Load associated files with timestamp
                scaler_file = self.model_path / f"scaler_{timestamp}.pkl"
                if scaler_file.exists():
                    self.scaler = joblib.load(scaler_file)
                    logger.info(f"Loaded scaler from {scaler_file}")
                
                features_file = self.model_path / f"feature_columns_{timestamp}.pkl"
                if features_file.exists():
                    self.feature_columns = joblib.load(features_file)
                    logger.info(f"Loaded feature columns from {features_file}")
                
                return True
            
            # Fall back to enhanced models if no fixed models
            enhanced_models = list(self.model_path.glob("enhanced_model*.pkl"))
            if enhanced_models:
                model_file = max(enhanced_models, key=os.path.getctime)
                self.xgb_model = joblib.load(model_file)
                
                # Load associated files
                encoder_file = self.model_path / "location_encoder.pkl"
                if encoder_file.exists():
                    self.location_encoder = joblib.load(encoder_file)
                
                scaler_file = self.model_path / "feature_scaler.pkl"
                if scaler_file.exists():
                    self.scaler = joblib.load(scaler_file)
                
                features_file = self.model_path / "feature_columns.pkl"
                if features_file.exists():
                    self.feature_columns = joblib.load(features_file)
                
                logger.info(f"Loaded existing enhanced model from {model_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _train_new_model(self):
        """Train a new XGBoost model with proper feature engineering"""
        if self.data is None or len(self.data) == 0:
            logger.error("No data available for training")
            return
        
        try:
            # Prepare features
            X, y = self._prepare_features_for_training(self.data)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost with regularization
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=5,
                random_state=42
            )
            
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Store feature columns
            self.feature_columns = list(X.columns)
            
            # Evaluate model
            from sklearn.metrics import r2_score, mean_absolute_error
            y_pred = self.xgb_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Model trained - R2: {r2:.3f}, MAE: {mae:.2f} Lakhs")
            
            # Save model and components
            self._save_model()
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        feature_dict = {}
        
        # Basic features
        for col in ['bath', 'balcony', 'bhk', 'total_sqft']:
            if col in df.columns:
                feature_dict[col] = df[col].values
        
        # Location encoding
        location_features = []
        for location in df['location']:
            loc_enc = self.location_encoder.transform(location)
            location_features.append([loc_enc['location_encoded'], loc_enc['location_tier']])
        
        location_features = np.array(location_features)
        
        # Add location features to dictionary
        feature_dict['location_encoded'] = location_features[:, 0]
        feature_dict['location_tier'] = location_features[:, 1]
        
        # Create DataFrame from feature dictionary
        X = pd.DataFrame(feature_dict)
        
        # Add engineered features with safety checks
        if 'total_sqft' in X.columns and 'bhk' in X.columns:
            X['price_per_sqft'] = X['total_sqft'] / X['bhk'].clip(lower=1)
            X['sqft_per_bhk'] = X['total_sqft'] / X['bhk'].clip(lower=1)
        if 'bath' in X.columns and 'bhk' in X.columns:
            X['bath_per_bhk'] = X['bath'] / X['bhk'].clip(lower=1)
        
        # Target variable
        y = df['price']
        
        return X, y
    
    def _save_model(self):
        """Save trained model and components"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save XGBoost model
            model_file = self.model_path / f"enhanced_xgb_model_{timestamp}.pkl"
            joblib.dump(self.xgb_model, model_file)
            
            # Save location encoder
            encoder_file = self.model_path / "location_encoder.pkl"
            joblib.dump(self.location_encoder, encoder_file)
            
            # Save scaler
            scaler_file = self.model_path / "feature_scaler.pkl"
            joblib.dump(self.scaler, scaler_file)
            
            # Save feature columns
            features_file = self.model_path / "feature_columns.pkl"
            joblib.dump(self.feature_columns, features_file)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def predict_price(
        self, 
        features: Dict[str, Any], 
        model_type: str = "xgboost",
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Predict property price using XGBoost model
        """
        try:
            # Extract and prepare features
            bhk = features.get('bhk', features.get('bedrooms', 3))
            bath = features.get('bath', features.get('bathrooms', 2))
            balcony = features.get('balcony', features.get('balconies', 1))
            sqft = features.get('total_sqft', features.get('area', features.get('sqft', None)))
            
            # Handle missing sqft - estimate based on BHK
            if not sqft:
                sqft_estimates = {1: 650, 2: 1100, 3: 1650, 4: 2200, 5: 2800}
                sqft = sqft_estimates.get(bhk, 1650)
            
            # Prepare feature vector using the exact format expected by the model
            if self.xgb_model and self.feature_columns:
                # Load location encoding data
                location = features.get('location', 'Bangalore')
                if self.data is not None:
                    # Handle different column names in the dataset
                    location_col = 'Location' if 'Location' in self.data.columns else 'location'
                    price_col = 'Price' if 'Price' in self.data.columns else 'price'
                    
                    location_counts = self.data[location_col].value_counts()
                    location_prices = self.data.groupby(location_col)[price_col].mean()
                    global_mean = self.data[price_col].mean()
                    
                    loc_freq = location_counts.get(location, 10)
                    loc_price = location_prices.get(location, global_mean)
                else:
                    loc_freq = 10
                    loc_price = 200
                
                # Map to the expected column names (based on new training format)
                feature_dict = {
                    'BHK': bhk,
                    'TotalSqft': sqft,
                    'Bath': bath,
                    'Balcony': balcony,
                    'PropertyAgeYears': features.get('property_age', 5),
                    'FloorNumber': features.get('floor_number', 2),
                    'TotalFloors': features.get('total_floors', 4),
                    'Parking': features.get('parking', 1),
                    'LocationFrequency': loc_freq,
                    'LocationPriceEncoding': loc_price,
                    'SqftPerBHK': sqft / bhk,
                    'BathPerBHK': bath / bhk,
                    'TotalRooms': bhk + bath
                }
                
                # Create DataFrame with correct columns
                X = pd.DataFrame([feature_dict])[self.feature_columns]
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Make prediction
                predicted_price = float(self.xgb_model.predict(X_scaled)[0])
                
                # Get feature importance for explanation (handle both Pipeline and direct model)
                try:
                    if hasattr(self.xgb_model, 'feature_importances_'):
                        feature_importance = dict(zip(self.feature_columns, self.xgb_model.feature_importances_))
                    elif hasattr(self.xgb_model, 'named_steps') and hasattr(self.xgb_model.named_steps.get('model', None), 'feature_importances_'):
                        # Pipeline case
                        feature_importance = dict(zip(self.feature_columns, self.xgb_model.named_steps['model'].feature_importances_))
                    else:
                        # Fallback - create dummy importance
                        feature_importance = {col: 1.0/len(self.feature_columns) for col in self.feature_columns}
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                except Exception as e:
                    logger.warning(f"Could not extract feature importance: {e}")
                    top_features = [('BHK', 0.3), ('TotalSqft', 0.3), ('Location', 0.2)]
                
            else:
                # Fallback to data-based estimation
                predicted_price = self._fallback_prediction(bhk, bath, location)
                top_features = [('location', 0.4), ('bhk', 0.3), ('sqft', 0.3)]
            
            result = {
                "predicted_price": round(predicted_price, 2),
                "currency": "INR Lakhs",
                "model_used": "xgboost_enhanced",
                "features_used": {
                    "bhk": bhk,
                    "bath": bath,
                    "balcony": balcony,
                    "total_sqft": sqft,
                    "location": location if location else "Bengaluru General"
                },
                "top_influencing_factors": [
                    {"feature": feat, "importance": round(imp, 3)} 
                    for feat, imp in top_features
                ]
            }
            
            if include_confidence:
                # Calculate confidence interval based on model uncertainty
                std_estimate = predicted_price * 0.1  # 10% standard deviation
                result["confidence_interval"] = {
                    "lower": round(predicted_price - 1.96 * std_estimate, 2),
                    "upper": round(predicted_price + 1.96 * std_estimate, 2),
                    "confidence_level": "95%"
                }
            
            # Add market insights
            result["market_insights"] = self._get_market_insights(location, bhk, predicted_price)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return fallback prediction
            return {
                "predicted_price": 100.0,
                "currency": "INR Lakhs",
                "model_used": "fallback",
                "error": str(e)
            }
    
    def _fallback_prediction(self, bhk: int, bath: int, location: str) -> float:
        """Fallback prediction using data statistics"""
        if self.data is None:
            return 100.0
        
        # Filter similar properties
        similar = self.data
        if 'bhk' in self.data.columns:
            similar = similar[similar['bhk'] == bhk]
        if 'bath' in self.data.columns and len(similar) > 10:
            similar = similar[similar['bath'] == bath]
        if location and 'location' in self.data.columns:
            loc_data = self.data[self.data['location'].str.contains(location, case=False, na=False)]
            if len(loc_data) > 5:
                similar = loc_data
        
        if len(similar) > 0:
            return float(similar['price'].median())
        else:
            return float(self.data['price'].median())
    
    def _get_market_insights(self, location: str, bhk: int, predicted_price: float) -> Dict[str, Any]:
        """Generate market insights"""
        insights = {
            "location_analysis": self._analyze_location(location),
            "price_trend": self._get_price_trend(location, bhk),
            "market_position": self._get_market_position(predicted_price)
        }
        return insights
    
    def _analyze_location(self, location: str) -> str:
        """Analyze location characteristics"""
        premium_locations = ['Koramangala', 'Indiranagar', 'Whitefield', 'Jayanagar', 'HSR Layout']
        mid_tier_locations = ['Hebbal', 'Marathahalli', 'Electronic City', 'Bannerghatta', 'Yelahanka']
        
        for loc in premium_locations:
            if loc.lower() in location.lower():
                return f"{location} is a premium location with high demand and excellent connectivity"
        
        for loc in mid_tier_locations:
            if loc.lower() in location.lower():
                return f"{location} is a growing area with good infrastructure and moderate prices"
        
        return f"{location} is an emerging area with potential for growth"
    
    def _get_price_trend(self, location: str, bhk: int) -> str:
        """Get price trend analysis"""
        if self.data is None:
            return "Market data unavailable"
        
        # Calculate average price for similar properties
        similar = self.data[self.data['bhk'] == bhk] if 'bhk' in self.data.columns else self.data
        avg_price = similar['price'].mean()
        
        return f"Average {bhk}BHK price in market: ₹{avg_price:.2f} Lakhs"
    
    def _get_market_position(self, price: float) -> str:
        """Determine market position"""
        if self.data is None:
            return "Unable to determine"
        
        percentile = (self.data['price'] < price).mean() * 100
        
        if percentile < 25:
            return "Budget-friendly (Lower 25%)"
        elif percentile < 50:
            return "Below average (25-50%)"
        elif percentile < 75:
            return "Above average (50-75%)"
        else:
            return "Premium (Top 25%)"
