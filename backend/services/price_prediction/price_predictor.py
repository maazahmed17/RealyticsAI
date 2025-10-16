"""
Price Prediction Service
Handles ML model operations, predictions, and market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pickle
import os
from pathlib import Path
import asyncio
from datetime import datetime
import logging

# Import from your existing price prediction system
from .run_local_data_pipeline import analyze_data, create_custom_pipeline
from .csv_data_ingestor import CSVDataIngestor, ExcelDataIngestor

logger = logging.getLogger(__name__)


class PricePredictionService:
    """
    Price Prediction Service for RealyticsAI Platform
    Manages ML models, predictions, and market analysis
    """
    
    def __init__(self):
        self.models = {}
        self.data_cache = {}
        self.default_data_path = "/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv"
        self.model_storage_path = "../../data/models"
        
        # Initialize with default data
        asyncio.create_task(self._initialize_service())
    
    async def _initialize_service(self):
        """Initialize the service with default models and data"""
        try:
            # Load default Bengaluru dataset
            await self._load_default_data()
            logger.info("Price Prediction Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Price Prediction Service: {e}")
    
    async def _load_default_data(self):
        """Load the default Bengaluru housing dataset"""
        try:
            if os.path.exists(self.default_data_path):
                df = pd.read_csv(self.default_data_path)
                self.data_cache['bengaluru'] = df
                logger.info(f"Loaded Bengaluru dataset: {df.shape[0]} properties")
            else:
                logger.warning(f"Default data file not found: {self.default_data_path}")
        except Exception as e:
            logger.error(f"Error loading default data: {e}")
    
    async def predict_price(
        self, 
        features: Dict[str, Any], 
        model_type: str = "local",
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Predict property price based on features
        """
        try:
            # For now, use the Bengaluru dataset for predictions
            if 'bengaluru' in self.data_cache:
                df = self.data_cache['bengaluru']
                
                # Extract relevant features
                bath = features.get('bath', features.get('bathrooms', 2))
                balcony = features.get('balcony', features.get('balconies', 1))
                
                # Find similar properties for price estimation
                similar_properties = df[
                    (df['bath'] == bath) & 
                    (df['balcony'] == balcony)
                ]
                
                if len(similar_properties) > 0:
                    # Calculate price statistics
                    avg_price = similar_properties['price'].mean()
                    min_price = similar_properties['price'].min()
                    max_price = similar_properties['price'].max()
                    std_price = similar_properties['price'].std()
                    
                    # Use average as prediction
                    predicted_price = round(avg_price, 2)
                    
                    result = {
                        "predicted_price": predicted_price,
                        "currency": "INR Lakhs",
                        "model_used": f"local_bengaluru_similar_properties",
                        "similar_properties": similar_properties.head(5).to_dict('records')
                    }
                    
                    if include_confidence:
                        result["confidence_interval"] = {
                            "lower": round(max(avg_price - std_price, min_price), 2),
                            "upper": round(min(avg_price + std_price, max_price), 2),
                            "std_dev": round(std_price, 2)
                        }
                    
                    # Add market insights
                    result["market_insights"] = {
                        "similar_properties_count": len(similar_properties),
                        "price_range": {
                            "min": round(min_price, 2),
                            "max": round(max_price, 2),
                            "average": round(avg_price, 2)
                        },
                        "market_position": self._get_market_position(predicted_price, df['price'])
                    }
                    
                    return result
                else:
                    # If no exact matches, use overall statistics
                    overall_avg = df['price'].mean()
                    
                    return {
                        "predicted_price": round(overall_avg, 2),
                        "currency": "INR Lakhs", 
                        "model_used": "local_bengaluru_fallback",
                        "market_insights": {
                            "note": "No exact matches found, using market average",
                            "market_average": round(overall_avg, 2)
                        }
                    }
            
            else:
                # Fallback prediction if no data available
                return {
                    "predicted_price": 85.0,  # Default reasonable price for Bengaluru
                    "currency": "INR Lakhs",
                    "model_used": "fallback_estimate",
                    "market_insights": {
                        "note": "Using fallback estimation due to data unavailability"
                    }
                }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def _get_market_position(self, price: float, market_prices: pd.Series) -> str:
        """Determine market position of the predicted price"""
        percentile = (market_prices < price).mean() * 100
        
        if percentile < 25:
            return "budget_friendly"
        elif percentile < 50:
            return "below_average"
        elif percentile < 75:
            return "above_average"
        else:
            return "premium"
    
    async def train_model(
        self, 
        data_file_path: str, 
        target_column: str,
        model_name: str = "custom_model"
    ) -> Dict[str, Any]:
        """
        Train a new model with provided data
        """
        try:
            # Load and analyze data
            df, potential_targets = analyze_data(data_file_path)
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # For now, store the training data for similar property matching
            self.data_cache[model_name] = df
            
            # Calculate basic statistics
            target_stats = df[target_column].describe()
            
            return {
                "model_name": model_name,
                "data_shape": df.shape,
                "target_column": target_column,
                "target_statistics": target_stats.to_dict(),
                "training_completed": datetime.now().isoformat(),
                "status": "completed"
            }
        
        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise Exception(f"Model training failed: {str(e)}")
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        
        for model_name, data in self.data_cache.items():
            models.append({
                "model_name": model_name,
                "status": "active",
                "last_trained": datetime.now().isoformat(),
                "performance_metrics": {
                    "data_points": len(data) if isinstance(data, pd.DataFrame) else 0
                },
                "features_expected": ["bath", "balcony"]
            })
        
        return models
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific model"""
        if model_name not in self.data_cache:
            raise ValueError(f"Model {model_name} not found")
        
        data = self.data_cache[model_name]
        
        return {
            "model_name": model_name,
            "status": "active",
            "last_trained": datetime.now().isoformat(),
            "performance_metrics": {
                "data_points": len(data),
                "features": list(data.columns) if hasattr(data, 'columns') else []
            },
            "features_expected": ["bath", "balcony"]
        }
    
    async def get_market_analysis(
        self, 
        location: str, 
        property_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get market analysis for a location"""
        try:
            if 'bengaluru' in self.data_cache:
                df = self.data_cache['bengaluru']
                
                # Filter by location if available
                location_data = df[df['location'].str.contains(location, case=False, na=False)]
                
                if len(location_data) == 0:
                    location_data = df  # Use all data if location not found
                
                analysis = {
                    "location": location,
                    "total_properties": len(location_data),
                    "price_statistics": {
                        "average": round(location_data['price'].mean(), 2),
                        "median": round(location_data['price'].median(), 2),
                        "min": round(location_data['price'].min(), 2),
                        "max": round(location_data['price'].max(), 2),
                        "std_dev": round(location_data['price'].std(), 2)
                    },
                    "property_types": location_data.groupby('size').size().to_dict(),
                    "market_trends": {
                        "price_per_bathroom": location_data.groupby('bath')['price'].mean().to_dict(),
                        "price_per_balcony": location_data.groupby('balcony')['price'].mean().to_dict()
                    }
                }
                
                return analysis
            
            else:
                return {
                    "location": location,
                    "error": "Market data not available",
                    "status": "data_unavailable"
                }
        
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            raise Exception(f"Market analysis failed: {str(e)}")
    
    async def find_similar_properties(
        self,
        bedrooms: int,
        bathrooms: int,
        location: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar properties for comparison"""
        try:
            if 'bengaluru' in self.data_cache:
                df = self.data_cache['bengaluru']
                
                # Filter by criteria
                query_df = df[df['bath'] == bathrooms]
                
                if location:
                    query_df = query_df[
                        query_df['location'].str.contains(location, case=False, na=False)
                    ]
                
                # Get top similar properties
                similar = query_df.head(max_results)
                
                return similar.to_dict('records')
            
            else:
                return []
        
        except Exception as e:
            logger.error(f"Similar properties search error: {e}")
            return []
