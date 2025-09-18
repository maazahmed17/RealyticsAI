"""
Property Recommendation Engine for RealyticsAI
==============================================
Main engine that orchestrates property recommendations with different models.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from .data_loader import PropertyDataLoader
from .models.simple_recommender import SimpleContentRecommender
from .query_analyzer import QueryAnalyzer, QueryIntent

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Main recommendation engine for RealyticsAI"""
    
    def __init__(self):
        """Initialize the recommendation engine"""
        
        logger.info("Initializing RealyticsAI Recommendation Engine")
        
        # Initialize components
        self.data_loader = PropertyDataLoader()
        self.query_analyzer = QueryAnalyzer()
        
        # Load data and initialize models
        self.property_data = None
        self.simple_recommender = None
        
        # Initialize the system
        self._initialize_models()
        
        logger.info("Recommendation Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize all recommendation models"""
        try:
            # Load property data
            logger.info("Loading property data...")
            self.property_data = self.data_loader.load_properties()
            logger.info(f"Loaded {len(self.property_data)} properties")
            
            # Initialize TF-IDF recommender
            logger.info("Initializing TF-IDF recommender...")
            self.simple_recommender = SimpleContentRecommender(self.property_data)
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def get_recommendations(self, 
                          query: str, 
                          top_k: int = 10, 
                          user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get property recommendations based on user query
        
        Args:
            query: User's search query
            top_k: Number of recommendations to return
            user_context: Optional user context (preferences, history, etc.)
            
        Returns:
            Dictionary containing recommendations and metadata
        """
        try:
            logger.info(f"Processing recommendation request: '{query}'")
            
            # Analyze the query
            analysis = self.query_analyzer.analyze_query(query)
            
            # Extract search filters
            filters = self.query_analyzer.extract_search_filters(query)
            
            # Get recommendations using TF-IDF model (for now)
            recommendations = self.simple_recommender.recommend(
                query_text=query,
                top_k=top_k,
                filters=filters
            )
            
            # Process recommendations for response
            processed_recs = self._process_recommendations(recommendations)
            
            result = {
                "success": True,
                "query": query,
                "query_analysis": {
                    "intent": analysis["intent"].value,
                    "confidence": analysis["confidence"],
                    "extracted_filters": filters
                },
                "recommendations": processed_recs,
                "total_found": len(processed_recs),
                "model_used": "TF-IDF Simple Recommender",
                "response_time_ms": 0  # TODO: Add timing
            }
            
            logger.info(f"Generated {len(processed_recs)} recommendations")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "recommendations": []
            }
    
    def get_similar_properties(self, property_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Find properties similar to a given property
        
        Args:
            property_id: ID of the reference property
            top_k: Number of similar properties to return
            
        Returns:
            Dictionary containing similar properties
        """
        try:
            # Get the reference property
            reference_property = self.data_loader.get_property_by_id(property_id)
            if not reference_property:
                return {
                    "success": False,
                    "error": f"Property {property_id} not found",
                    "similar_properties": []
                }
            
            # Get similar properties
            similar_props = self.simple_recommender.get_similar_properties(property_id, top_k)
            processed_similar = self._process_recommendations(similar_props)
            
            return {
                "success": True,
                "reference_property": self._process_single_property(reference_property),
                "similar_properties": processed_similar,
                "total_found": len(processed_similar),
                "model_used": "TF-IDF Similarity"
            }
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {e}")
            return {
                "success": False,
                "error": str(e),
                "similar_properties": []
            }
    
    def get_recommendations_by_features(self, 
                                      features: Dict[str, Any], 
                                      top_k: int = 10) -> Dict[str, Any]:
        """Get recommendations based on specific property features
        
        Args:
            features: Dictionary of desired property features
            top_k: Number of recommendations to return
            
        Returns:
            Dictionary containing feature-based recommendations
        """
        try:
            recommendations = self.simple_recommender.get_recommendations_by_features(features, top_k)
            processed_recs = self._process_recommendations(recommendations)
            
            return {
                "success": True,
                "requested_features": features,
                "recommendations": processed_recs,
                "total_found": len(processed_recs),
                "model_used": "Feature-based TF-IDF"
            }
            
        except Exception as e:
            logger.error(f"Error generating feature-based recommendations: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": []
            }
    
    def _process_recommendations(self, recommendations_df) -> List[Dict[str, Any]]:
        """Process raw recommendations DataFrame for API response
        
        Args:
            recommendations_df: DataFrame with recommendation results
            
        Returns:
            List of processed property dictionaries
        """
        if recommendations_df.empty:
            return []
        
        processed = []
        
        for _, row in recommendations_df.iterrows():
            property_dict = {
                "property_id": row.get("PropertyID", f"PROP_{row.name}"),
                "location": row.get("location", "Unknown Location"),
                "size": row.get("size", "N/A"),
                "total_sqft": float(row.get("total_sqft", 0)) if pd.notna(row.get("total_sqft")) else None,
                "price": float(row.get("price", 0)) if pd.notna(row.get("price")) else None,
                "price_per_sqft": None,
                "similarity_score": float(row.get("similarity", 0)) if pd.notna(row.get("similarity")) else 0.0,
                "area_type": row.get("area_type", "Unknown"),
                "bathrooms": float(row.get("bathroom", 0)) if pd.notna(row.get("bathroom")) else None,
                "balcony": float(row.get("balcony", 0)) if pd.notna(row.get("balcony")) else None
            }
            
            # Calculate price per sqft if both values are available
            if property_dict["price"] and property_dict["total_sqft"] and property_dict["total_sqft"] > 0:
                property_dict["price_per_sqft"] = property_dict["price"] / property_dict["total_sqft"]
            
            # Format price in lakhs for display
            if property_dict["price"]:
                property_dict["price_lakhs"] = property_dict["price"] / 100000
            
            processed.append(property_dict)
        
        return processed
    
    def _process_single_property(self, property_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single property dictionary
        
        Args:
            property_dict: Raw property data
            
        Returns:
            Processed property dictionary
        """
        import pandas as pd
        
        processed = {
            "property_id": property_dict.get("PropertyID", "Unknown"),
            "location": property_dict.get("location", "Unknown Location"),
            "size": property_dict.get("size", "N/A"),
            "total_sqft": float(property_dict.get("total_sqft", 0)) if pd.notna(property_dict.get("total_sqft")) else None,
            "price": float(property_dict.get("price", 0)) if pd.notna(property_dict.get("price")) else None,
            "area_type": property_dict.get("area_type", "Unknown"),
            "bathrooms": float(property_dict.get("bathroom", 0)) if pd.notna(property_dict.get("bathroom")) else None,
            "balcony": float(property_dict.get("balcony", 0)) if pd.notna(property_dict.get("balcony")) else None
        }
        
        # Calculate derived values
        if processed["price"] and processed["total_sqft"] and processed["total_sqft"] > 0:
            processed["price_per_sqft"] = processed["price"] / processed["total_sqft"]
        
        if processed["price"]:
            processed["price_lakhs"] = processed["price"] / 100000
        
        return processed
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the recommendation system
        
        Returns:
            Dictionary with system statistics
        """
        try:
            data_summary = self.data_loader.get_data_summary()
            recommender_stats = self.simple_recommender.get_stats()
            
            return {
                "status": "operational",
                "data_statistics": data_summary,
                "model_statistics": {
                    "simple_recommender": recommender_stats
                },
                "supported_features": {
                    "text_search": True,
                    "location_filter": True,
                    "bhk_filter": True,
                    "price_filter": True,
                    "sqft_filter": True,
                    "similarity_search": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the recommendation system
        
        Returns:
            Dictionary with health status
        """
        try:
            # Check if models are loaded
            models_loaded = {
                "data_loader": self.data_loader is not None,
                "property_data": self.property_data is not None,
                "simple_recommender": self.simple_recommender is not None,
                "query_analyzer": self.query_analyzer is not None
            }
            
            all_healthy = all(models_loaded.values())
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "models_loaded": models_loaded,
                "total_properties": len(self.property_data) if self.property_data is not None else 0,
                "system_ready": all_healthy
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "system_ready": False
            }