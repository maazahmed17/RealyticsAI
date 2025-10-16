"""
Property Comparables Finder Module
===================================
Finds truly comparable properties based on location, BHK, and size criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class ComparablesFinder:
    """Find comparable properties for accurate market analysis."""
    
    def __init__(self, data: pd.DataFrame = None):
        """Initialize with property data.
        
        Args:
            data: DataFrame containing property data
        """
        self.data = data
        if self.data is None:
            self.load_data()
    
    def load_data(self):
        """Load property data from CSV file."""
        # Use the correct data path
        data_path = Path("/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv")
        if data_path.exists():
            self.data = pd.read_csv(data_path)
            self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis."""
        if self.data is None:
            return
        
        # Clean total_sqft column
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
        
        # Extract BHK from size column
        if 'size' in self.data.columns and 'bhk' not in self.data.columns:
            self.data['bhk'] = self.data['size'].str.extract('(\\d+)').astype(float)
        
        # Clean location names (trim whitespace, lowercase for matching)
        if 'location' in self.data.columns:
            self.data['location_clean'] = self.data['location'].str.strip().str.lower()
    
    def find_comparables(self, 
                        location: str, 
                        bhk: int, 
                        sqft: float,
                        strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find comparable properties based on refined criteria.
        
        Args:
            location: Property location
            bhk: Number of bedrooms
            sqft: Total square footage
            strict: If True, use strict matching criteria
            
        Returns:
            Tuple of (comparable_properties_df, search_stats_dict)
        """
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame(), {"count": 0, "criteria": "No data available"}
        
        comparables = self.data.copy()
        location_lower = location.strip().lower() if location else ""
        
        # Track search criteria for transparency
        search_criteria = []
        
        # 1. Location matching (exact match in strict mode)
        if location and 'location_clean' in comparables.columns:
            if strict:
                # Exact location match
                location_matches = comparables['location_clean'] == location_lower
                search_criteria.append(f"Location: {location} (exact match)")
            else:
                # Partial match for fallback
                location_matches = comparables['location_clean'].str.contains(
                    location_lower, case=False, na=False
                )
                search_criteria.append(f"Location: {location} (partial match)")
            
            comparables = comparables[location_matches]
        
        # 2. BHK matching (±1 in strict mode, ±2 otherwise)
        if bhk and 'bhk' in comparables.columns and len(comparables) > 0:
            bhk_tolerance = 1 if strict else 2
            bhk_matches = (comparables['bhk'] >= bhk - bhk_tolerance) & \
                         (comparables['bhk'] <= bhk + bhk_tolerance)
            comparables = comparables[bhk_matches]
            search_criteria.append(f"BHK: {bhk} ± {bhk_tolerance}")
        
        # 3. Size matching (±20% in strict mode, ±30% otherwise)
        if sqft and sqft > 0 and 'total_sqft' in comparables.columns and len(comparables) > 0:
            size_tolerance = 0.2 if strict else 0.3
            min_sqft = sqft * (1 - size_tolerance)
            max_sqft = sqft * (1 + size_tolerance)
            size_matches = (comparables['total_sqft'] >= min_sqft) & \
                          (comparables['total_sqft'] <= max_sqft)
            comparables = comparables[size_matches]
            search_criteria.append(f"Size: {sqft:.0f} sqft ± {int(size_tolerance*100)}%")
        
        # If strict search returns too few results, try relaxed criteria
        if strict and len(comparables) < 5:
            return self.find_comparables(location, bhk, sqft, strict=False)
        
        # Calculate statistics
        stats = {
            "count": len(comparables),
            "criteria": " | ".join(search_criteria),
            "search_mode": "strict" if strict else "relaxed"
        }
        
        if len(comparables) > 0 and 'price' in comparables.columns:
            stats.update({
                "avg_price": comparables['price'].mean(),
                "median_price": comparables['price'].median(),
                "min_price": comparables['price'].min(),
                "max_price": comparables['price'].max(),
                "price_std": comparables['price'].std(),
                "price_25_percentile": comparables['price'].quantile(0.25),
                "price_75_percentile": comparables['price'].quantile(0.75)
            })
        
        return comparables, stats
    
    def get_market_context(self, 
                          location: str = None,
                          bhk: int = None) -> Dict[str, Any]:
        """Get broader market context for comparison.
        
        Args:
            location: Property location (optional)
            bhk: Number of bedrooms (optional)
            
        Returns:
            Dictionary with market context information
        """
        context = {}
        
        if self.data is None or len(self.data) == 0:
            return context
        
        # Overall market stats
        if 'price' in self.data.columns:
            context['overall_market'] = {
                'avg_price': self.data['price'].mean(),
                'median_price': self.data['price'].median(),
                'total_properties': len(self.data)
            }
        
        # Location-specific stats
        if location and 'location_clean' in self.data.columns:
            location_data = self.data[
                self.data['location_clean'].str.contains(
                    location.lower(), case=False, na=False
                )
            ]
            if len(location_data) > 0 and 'price' in location_data.columns:
                context['location_market'] = {
                    'avg_price': location_data['price'].mean(),
                    'median_price': location_data['price'].median(),
                    'property_count': len(location_data),
                    'location': location
                }
        
        # BHK-specific stats
        if bhk and 'bhk' in self.data.columns:
            bhk_data = self.data[self.data['bhk'] == bhk]
            if len(bhk_data) > 0 and 'price' in bhk_data.columns:
                context['bhk_market'] = {
                    'avg_price': bhk_data['price'].mean(),
                    'median_price': bhk_data['price'].median(),
                    'property_count': len(bhk_data),
                    'bhk': bhk
                }
        
        return context
