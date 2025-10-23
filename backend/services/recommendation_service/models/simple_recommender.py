"""
Simple Content-Based Recommender for RealyticsAI
=================================================
TF-IDF based property recommendation system.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

class SimpleContentRecommender:
    """TF-IDF based content recommender for properties"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the recommender with property data
        
        Args:
            df: DataFrame containing property data
        """
        self.df = df.reset_index(drop=True)
        logger.info(f"Initializing SimpleContentRecommender with {len(df)} properties")
        
        self.df['location_normalized'] = self.df['location'].str.lower().str.strip()
        logger.info("Created normalized location column for filtering.")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            ngram_range=(1, 2), 
            stop_words='english',
            max_features=5000  # Limit features for performance
        )
        
        # Create item matrix from search text
        search_texts = self.df["search_text"].fillna("")
        self.item_matrix = self.vectorizer.fit_transform(search_texts)
        
        # Create nearest neighbors model
        n_neighbors = min(50, len(df))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        self.nn.fit(self.item_matrix)
        
        logger.info("TF-IDF recommender initialized successfully")
    
    def recommend(self, 
                 query_text: str, 
                 top_k: int = 10, 
                 filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate property recommendations based on query
        
        Args:
            query_text: User's search query
            top_k: Number of recommendations to return
            filters: Optional filters to apply
            
        Returns:
            DataFrame with recommended properties and similarity scores
        """
        try:
            # Apply filters first
            filtered_df = self._apply_filters(filters) if filters else self.df.copy()
            
            if filtered_df.empty:
                logger.warning("No properties found after applying filters")
                return pd.DataFrame()
            
            # Get indices of filtered properties
            filtered_indices = filtered_df.index.values
            
            # Transform query to TF-IDF vector
            query_vec = self.vectorizer.transform([query_text or ""])
            
            # Find similar properties
            n_neighbors = min(top_k * 2, len(filtered_indices))  # Get more candidates
            distances, indices = self.nn.kneighbors(query_vec, n_neighbors=n_neighbors)
            
            # Filter results to only include filtered properties
            valid_results = []
            valid_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx in filtered_indices:
                    valid_results.append(idx)
                    valid_distances.append(distances[0][i])
                    
                if len(valid_results) >= top_k:
                    break
            
            if not valid_results:
                logger.warning("No matching properties found")
                return pd.DataFrame()
            
            # Get recommended properties
            recommendations = self.df.loc[valid_results].copy()
            recommendations["similarity"] = 1 - np.array(valid_distances)
            
            # Sort by similarity and return top-k
            result = recommendations.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
            
            logger.info(f"Generated {len(result)} recommendations for query: '{query_text}'")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return pd.DataFrame()
    
    def _apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the property dataset
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        if "location" in filters and filters["location"]:
            # 1. Normalize the user's input
            user_location_normalized = filters["location"].lower().strip()
            
            # 2. Filter on the pre-normalized column
            mask = filtered_df["location_normalized"] == user_location_normalized
            filtered_df = filtered_df[mask]
            logger.info(f"After location filter: {len(filtered_df)} properties")
        
        if "bhk" in filters and filters["bhk"]:
            mask = filtered_df["size"].str.contains(
                f"{filters['bhk']} BHK", case=False, na=False
            )
            filtered_df = filtered_df[mask]
        
        if "min_sqft" in filters and filters["min_sqft"]:
            filtered_df = filtered_df[filtered_df["total_sqft"] >= filters["min_sqft"]]
        
        if "max_price" in filters and filters["max_price"]:
            filtered_df = filtered_df[filtered_df["price"] <= filters["max_price"]]
        
        if "min_price" in filters and filters["min_price"]:
            filtered_df = filtered_df[filtered_df["price"] >= filters["min_price"]]
        
        if "min_balcony" in filters and filters["min_balcony"]:
            if "balcony" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["balcony"] >= filters["min_balcony"]]
        
        return filtered_df
    
    def get_similar_properties(self, property_id: str, top_k: int = 5) -> pd.DataFrame:
        """Find properties similar to a given property
        
        Args:
            property_id: ID of the reference property
            top_k: Number of similar properties to return
            
        Returns:
            DataFrame with similar properties
        """
        try:
            # Find the property
            property_mask = self.df["PropertyID"] == property_id
            if not property_mask.any():
                logger.warning(f"Property {property_id} not found")
                return pd.DataFrame()
            
            property_idx = self.df[property_mask].index[0]
            
            # Get the property's TF-IDF vector
            property_vec = self.item_matrix[property_idx:property_idx+1]
            
            # Find similar properties
            distances, indices = self.nn.kneighbors(property_vec, n_neighbors=top_k+1)
            
            # Remove the property itself from results
            similar_indices = [idx for idx in indices[0] if idx != property_idx][:top_k]
            similar_distances = [dist for i, dist in enumerate(distances[0]) if indices[0][i] != property_idx][:top_k]
            
            # Get similar properties
            similar_properties = self.df.loc[similar_indices].copy()
            similar_properties["similarity"] = 1 - np.array(similar_distances)
            
            return similar_properties.sort_values("similarity", ascending=False)
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {e}")
            return pd.DataFrame()
    
    def get_recommendations_by_features(self, 
                                      features: Dict[str, Any], 
                                      top_k: int = 10) -> pd.DataFrame:
        """Get recommendations based on specific property features
        
        Args:
            features: Dictionary of desired features
            top_k: Number of recommendations to return
            
        Returns:
            DataFrame with matching properties
        """
        # Convert features to a query string
        query_parts = []
        
        if "bhk" in features:
            query_parts.append(f"{features['bhk']} BHK")
        
        if "location" in features:
            query_parts.append(features["location"])
        
        if "area_type" in features:
            query_parts.append(features["area_type"])
        
        query_text = " ".join(query_parts)
        
        # Create filters from features
        filters = {}
        for key in ["bhk", "location", "min_sqft", "max_price", "min_price"]:
            if key in features:
                filters[key] = features[key]
        
        return self.recommend(query_text, top_k, filters)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the recommender
        
        Returns:
            Dictionary with recommender statistics
        """
        return {
            "total_properties": len(self.df),
            "tfidf_features": self.item_matrix.shape[1],
            "model_type": "TF-IDF + Cosine Similarity",
            "available_filters": ["location", "bhk", "min_sqft", "max_price", "min_price", "min_balcony"]
        }