"""
Natural Language Interface for Price Prediction
================================================
Integrates Gemini API to provide natural language interactions for price predictions.
"""

import json
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.config import settings
from backend.services.gemini_service import get_gemini_service
from backend.services.price_prediction.feature_engineering_advanced import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)


class PricePredictionNLP:
    """Natural language interface for price prediction"""
    
    def __init__(self):
        """Initialize the NLP interface"""
        self.gemini = get_gemini_service()
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.data = None
        
        # Load model and data
        self._load_model()
        self._load_data()
        
        logger.info("Price Prediction NLP interface initialized")
    
    def _load_model(self):
        """Load the trained model and related files"""
        try:
            if settings.PRICE_MODEL_PATH and settings.PRICE_MODEL_PATH.exists():
                self.model = joblib.load(settings.PRICE_MODEL_PATH)
                logger.info(f"Model loaded from {settings.PRICE_MODEL_PATH}")
            
            if settings.SCALER_PATH and settings.SCALER_PATH.exists():
                self.scaler = joblib.load(settings.SCALER_PATH)
                logger.info(f"Scaler loaded from {settings.SCALER_PATH}")
            
            if settings.FEATURE_LIST_PATH and settings.FEATURE_LIST_PATH.exists():
                with open(settings.FEATURE_LIST_PATH, 'r') as f:
                    self.features = [line.strip() for line in f.readlines()]
                logger.info(f"Features loaded: {len(self.features)} features")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def _load_data(self):
        """Load the Bengaluru dataset for context"""
        try:
            self.data = pd.read_csv(settings.BENGALURU_DATA_PATH)
            
            # Clean total_sqft if needed
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
            
            # Extract BHK from size if needed
            if 'size' in self.data.columns and 'bhk' not in self.data.columns:
                self.data['bhk'] = self.data['size'].str.extract('(\d+)').astype(float)
            
            logger.info(f"Data loaded: {len(self.data)} properties")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about price prediction
        
        Parameters:
        -----------
        query : str
            Natural language query from user
            
        Returns:
        --------
        dict : Response with prediction and explanation
        """
        try:
            # Parse the query to extract property features
            features = self._extract_features_from_query(query)
            
            if not features:
                # If no specific features, provide general guidance
                return self._handle_general_query(query)
            
            # Make prediction
            prediction_result = self._predict_price(features)
            
            # Get market context
            market_context = self._get_market_context(features)
            
            # Generate natural language response
            response = self._generate_nl_response(query, features, prediction_result, market_context)
            
            return {
                "success": True,
                "query": query,
                "extracted_features": features,
                "prediction": prediction_result,
                "market_context": market_context,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": self._generate_error_response(str(e))
            }
    
    def _extract_features_from_query(self, query: str) -> Dict[str, Any]:
        """Extract property features from natural language query"""
        
        prompt = f"""
        Extract property features from the following query. Return a JSON object with these fields:
        - bhk (number of bedrooms)
        - bath (number of bathrooms)
        - balcony (number of balconies)
        - total_sqft (total square feet)
        - location (location name)
        - area_type (if mentioned)
        - availability (if mentioned)
        
        If a field is not mentioned, omit it from the JSON.
        
        Query: {query}
        
        Return ONLY the JSON object, no additional text.
        """
        
        response = self.gemini.generate_response(prompt)
        
        try:
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group())
                
                # Set defaults for missing required fields
                defaults = {
                    'bhk': 2,
                    'bath': 2,
                    'balcony': 1,
                    'total_sqft': 1200
                }
                
                for key, default in defaults.items():
                    if key not in features:
                        features[key] = default
                
                return features
            
        except Exception as e:
            logger.warning(f"Could not extract features: {e}")
        
        return {}
    
    def _predict_price(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction based on features"""
        
        if not self.model:
            return {"error": "Model not loaded"}
        
        try:
            # Create a DataFrame with the features
            df = pd.DataFrame([features])
            
            # Apply feature engineering
            df_engineered = self.feature_engineer.transform(df)
            
            # Select only the features used in training
            if self.features:
                # Get numeric features
                numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
                
                # Use features that exist in both
                available_features = [f for f in self.features if f in numeric_features]
                
                if not available_features:
                    # Use basic features
                    available_features = ['bath', 'balcony', 'total_sqft', 'bhk']
                    available_features = [f for f in available_features if f in df_engineered.columns]
                
                X = df_engineered[available_features].fillna(df_engineered[available_features].median())
            else:
                # Fallback to basic features
                X = df_engineered[['bath', 'balcony']].fillna(1)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Calculate confidence interval (simplified)
            confidence_range = prediction * 0.1  # ±10% confidence
            
            return {
                "predicted_price": float(prediction),
                "price_range": {
                    "min": float(prediction - confidence_range),
                    "max": float(prediction + confidence_range)
                },
                "confidence": "high" if confidence_range < prediction * 0.15 else "moderate",
                "currency": "INR Lakhs"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def _get_market_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get market context for the property"""
        
        if self.data is None:
            return {}
        
        context = {}
        
        try:
            # Filter similar properties
            similar = self.data.copy()
            
            if 'bhk' in features:
                similar = similar[similar['bhk'] == features['bhk']]
            if 'location' in features and 'location' in similar.columns:
                similar = similar[similar['location'].str.contains(features['location'], case=False, na=False)]
            
            if len(similar) > 0:
                context['similar_properties_count'] = len(similar)
                context['average_price'] = float(similar['price'].mean())
                context['median_price'] = float(similar['price'].median())
                context['price_range'] = {
                    'min': float(similar['price'].min()),
                    'max': float(similar['price'].max())
                }
                
                # Top locations
                if 'location' in similar.columns:
                    top_locations = similar['location'].value_counts().head(5)
                    context['popular_locations'] = top_locations.to_dict()
            
            # Overall market stats
            context['market_stats'] = {
                'total_properties': len(self.data),
                'average_price_overall': float(self.data['price'].mean()),
                'median_price_overall': float(self.data['price'].median())
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
        
        return context
    
    def _generate_nl_response(self, query: str, features: Dict[str, Any], 
                             prediction: Dict[str, Any], 
                             market_context: Dict[str, Any]) -> str:
        """Generate natural language response"""
        
        prompt = f"""
        Generate a helpful and informative response to this real estate query:
        
        User Query: {query}
        
        Property Features: {json.dumps(features, indent=2)}
        
        Prediction Results: {json.dumps(prediction, indent=2)}
        
        Market Context: {json.dumps(market_context, indent=2)}
        
        Please provide:
        1. A direct answer to the query
        2. The predicted price in a conversational format
        3. How this compares to similar properties
        4. Key factors affecting the price
        5. Investment insights or recommendations
        6. Any important considerations
        
        Make the response conversational, helpful, and easy to understand.
        Use Indian Rupees (₹) and mention prices in Lakhs.
        """
        
        return self.gemini.generate_response(prompt)
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries without specific property features"""
        
        # Get overall market insights
        market_stats = {}
        if self.data is not None:
            market_stats = {
                'average_price': float(self.data['price'].mean()),
                'median_price': float(self.data['price'].median()),
                'total_properties': len(self.data),
                'price_range': {
                    'min': float(self.data['price'].min()),
                    'max': float(self.data['price'].max())
                },
                'popular_locations': self.data['location'].value_counts().head(10).to_dict() if 'location' in self.data.columns else {}
            }
        
        response = self.gemini.answer_general_query(query, market_stats)
        
        return {
            "success": True,
            "query": query,
            "type": "general",
            "market_stats": market_stats,
            "response": response
        }
    
    def _generate_error_response(self, error: str) -> str:
        """Generate a helpful error response"""
        
        prompt = f"""
        Generate a helpful and apologetic response for this error:
        
        Error: {error}
        
        Provide:
        1. An apology for the issue
        2. A simple explanation of what might have gone wrong
        3. Suggestions for what the user can try instead
        4. An example of a valid query
        
        Keep it friendly and helpful.
        """
        
        return self.gemini.generate_response(prompt)
    
    def get_market_insights(self) -> str:
        """Generate comprehensive market insights"""
        
        if self.data is None:
            return "Market data not available."
        
        # Prepare market data
        market_data = {
            'total_properties': len(self.data),
            'price_statistics': {
                'mean': float(self.data['price'].mean()),
                'median': float(self.data['price'].median()),
                'std': float(self.data['price'].std()),
                'min': float(self.data['price'].min()),
                'max': float(self.data['price'].max())
            },
            'property_distribution': {
                'by_bhk': self.data['bhk'].value_counts().head(5).to_dict() if 'bhk' in self.data.columns else {},
                'by_location': self.data['location'].value_counts().head(10).to_dict() if 'location' in self.data.columns else {}
            },
            'trends': self._calculate_trends()
        }
        
        return self.gemini.generate_market_insights(market_data)
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate market trends"""
        
        trends = {}
        
        try:
            if 'bhk' in self.data.columns:
                # Price by BHK
                price_by_bhk = self.data.groupby('bhk')['price'].mean()
                trends['price_by_bhk'] = price_by_bhk.to_dict()
            
            if 'location' in self.data.columns:
                # Top appreciating locations (simplified)
                location_prices = self.data.groupby('location')['price'].agg(['mean', 'count'])
                top_locations = location_prices[location_prices['count'] >= 10].nlargest(5, 'mean')
                trends['top_expensive_locations'] = top_locations['mean'].to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
        
        return trends


def create_nlp_interface() -> PricePredictionNLP:
    """Factory function to create NLP interface"""
    return PricePredictionNLP()


__all__ = ["PricePredictionNLP", "create_nlp_interface"]
