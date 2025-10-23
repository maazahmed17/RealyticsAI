"""
Prediction Guardrails Module
============================
Implements sanity checks and outlier detection for property price predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionGuardrails:
    """Guardrails for detecting and handling prediction outliers."""
    
    def __init__(self, data_path: str = None):
        """Initialize guardrails with historical data.
        
        Args:
            data_path: Path to the historical property data CSV
        """
        if data_path is None:
            data_path = "/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv"
        
        self.data_path = Path(data_path)
        self.location_stats = {}
        self.global_stats = {}
        
        # Load and preprocess historical data
        self._load_historical_data()
        self._calculate_location_statistics()
    
    def _load_historical_data(self):
        """Load and preprocess historical property data."""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Clean sqft column - handle different column name variations
            sqft_col = None
            if 'TotalSqft' in self.data.columns:
                sqft_col = 'TotalSqft'
            elif 'total_sqft' in self.data.columns:
                sqft_col = 'total_sqft'
            
            if sqft_col:
                def convert_sqft(x):
                    try:
                        if '-' in str(x):
                            parts = str(x).split('-')
                            return (float(parts[0]) + float(parts[1])) / 2
                        return float(x)
                    except:
                        return np.nan
                self.data[sqft_col] = self.data[sqft_col].apply(convert_sqft)
                # Normalize column name for later use
                if sqft_col != 'total_sqft':
                    self.data['total_sqft'] = self.data[sqft_col]
            
            # Calculate price per sqft - handle different column name variations
            price_col = None
            if 'Price' in self.data.columns:
                price_col = 'Price'
            elif 'price' in self.data.columns:
                price_col = 'price'
            
            sqft_col = None
            if 'TotalSqft' in self.data.columns:
                sqft_col = 'TotalSqft'
            elif 'total_sqft' in self.data.columns:
                sqft_col = 'total_sqft'
            
            if price_col and sqft_col:
                self.data['price_per_sqft'] = (self.data[price_col] * 100000) / self.data[sqft_col]
            else:
                logger.error(f"Missing required columns. Available columns: {list(self.data.columns)}")
                self.data = pd.DataFrame()
                return
            
            # Remove obvious outliers (price_per_sqft < 1000 or > 50000)
            self.data = self.data[
                (self.data['price_per_sqft'] >= 1000) & 
                (self.data['price_per_sqft'] <= 50000)
            ]
            
            # Clean location names - handle different column name variations
            location_col = None
            if 'Location' in self.data.columns:
                location_col = 'Location'
            elif 'location' in self.data.columns:
                location_col = 'location'
            
            if location_col:
                self.data['location_clean'] = self.data[location_col].str.strip().str.lower()
            
            logger.info(f"Loaded {len(self.data)} properties for guardrail analysis")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.data = pd.DataFrame()
    
    def _calculate_location_statistics(self):
        """Calculate price per sqft statistics for each location."""
        if self.data.empty or 'location_clean' not in self.data.columns:
            return
        
        # Calculate global statistics
        self.global_stats = {
            'mean_price_per_sqft': self.data['price_per_sqft'].mean(),
            'median_price_per_sqft': self.data['price_per_sqft'].median(),
            'std_price_per_sqft': self.data['price_per_sqft'].std(),
            'min_price_per_sqft': self.data['price_per_sqft'].min(),
            'max_price_per_sqft': self.data['price_per_sqft'].max(),
            'percentile_95': self.data['price_per_sqft'].quantile(0.95),
            'percentile_5': self.data['price_per_sqft'].quantile(0.05)
        }
        
        # Calculate location-specific statistics
        for location in self.data['location_clean'].unique():
            loc_data = self.data[self.data['location_clean'] == location]
            
            if len(loc_data) >= 5:  # Only calculate stats if we have enough data
                self.location_stats[location] = {
                    'mean_price_per_sqft': loc_data['price_per_sqft'].mean(),
                    'median_price_per_sqft': loc_data['price_per_sqft'].median(),
                    'std_price_per_sqft': loc_data['price_per_sqft'].std(),
                    'min_price_per_sqft': loc_data['price_per_sqft'].min(),
                    'max_price_per_sqft': loc_data['price_per_sqft'].max(),
                    'sample_size': len(loc_data)
                }
        
        logger.info(f"Calculated statistics for {len(self.location_stats)} locations")
    
    def is_prediction_valid(self, 
                          prediction_price: float, 
                          sqft: float,
                          location: str = None,
                          threshold_multiplier: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
        """Check if a prediction is valid based on historical data.
        
        Args:
            prediction_price: Predicted price in Lakhs
            sqft: Property size in square feet
            location: Property location
            threshold_multiplier: Maximum allowed multiple of historical average (default: 3.0)
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        # Calculate predicted price per sqft
        if sqft <= 0:
            return True, {"reason": "Cannot validate without square footage", "status": "skipped"}
        
        predicted_price_per_sqft = (prediction_price * 100000) / sqft
        
        validation_details = {
            "predicted_price": prediction_price,
            "sqft": sqft,
            "predicted_price_per_sqft": predicted_price_per_sqft,
            "location": location,
            "threshold_multiplier": threshold_multiplier
        }
        
        # Get reference statistics
        if location:
            location_clean = location.strip().lower()
            
            # Try exact match first
            if location_clean in self.location_stats:
                stats = self.location_stats[location_clean]
                validation_details["comparison_type"] = "exact_location"
            else:
                # Try partial match
                matching_locations = [
                    loc for loc in self.location_stats 
                    if location_clean in loc or loc in location_clean
                ]
                
                if matching_locations:
                    # Use the first matching location
                    stats = self.location_stats[matching_locations[0]]
                    validation_details["comparison_type"] = "partial_location_match"
                    validation_details["matched_location"] = matching_locations[0]
                else:
                    # Fall back to global statistics
                    stats = self.global_stats
                    validation_details["comparison_type"] = "global"
        else:
            stats = self.global_stats
            validation_details["comparison_type"] = "global"
        
        # Add reference statistics to details
        validation_details["reference_mean"] = stats['mean_price_per_sqft']
        validation_details["reference_median"] = stats['median_price_per_sqft']
        validation_details["reference_max"] = stats.get('max_price_per_sqft', stats['mean_price_per_sqft'] * 2)
        
        # Check if prediction is within acceptable range
        upper_threshold = stats['mean_price_per_sqft'] * threshold_multiplier
        lower_threshold = stats['mean_price_per_sqft'] / threshold_multiplier
        
        validation_details["upper_threshold"] = upper_threshold
        validation_details["lower_threshold"] = lower_threshold
        
        # Determine if valid
        is_valid = lower_threshold <= predicted_price_per_sqft <= upper_threshold
        
        # Add outlier classification
        if not is_valid:
            if predicted_price_per_sqft > upper_threshold:
                validation_details["outlier_type"] = "too_high"
                validation_details["multiple_of_average"] = predicted_price_per_sqft / stats['mean_price_per_sqft']
            else:
                validation_details["outlier_type"] = "too_low"
                validation_details["fraction_of_average"] = predicted_price_per_sqft / stats['mean_price_per_sqft']
        else:
            validation_details["outlier_type"] = None
        
        validation_details["is_valid"] = is_valid
        
        return is_valid, validation_details
    
    def get_outlier_message(self, validation_details: Dict[str, Any]) -> str:
        """Generate appropriate message for outlier predictions.
        
        Args:
            validation_details: Details from is_prediction_valid check
            
        Returns:
            Professional message explaining the outlier
        """
        outlier_type = validation_details.get("outlier_type")
        location = validation_details.get("location", "this area")
        predicted_ppsf = validation_details.get("predicted_price_per_sqft", 0)
        reference_mean = validation_details.get("reference_mean", 0)
        
        if outlier_type == "too_high":
            multiple = validation_details.get("multiple_of_average", 0)
            
            if multiple > 5:
                return (
                    f"⚠️ **Extreme Outlier Detected**\n\n"
                    f"The estimated value for this property appears to be significantly higher "
                    f"than typical properties in {location} (over {multiple:.1f}x the area average).\n\n"
                    f"**Predicted**: ₹{predicted_ppsf:,.0f} per sq.ft\n"
                    f"**Area Average**: ₹{reference_mean:,.0f} per sq.ft\n\n"
                    f"This extreme variance could indicate:\n"
                    f"• Ultra-luxury specifications not captured in our model\n"
                    f"• Unique property features (penthouse, heritage value, etc.)\n"
                    f"• Data input errors\n\n"
                    f"**Recommendation**: Please consult with local real estate experts "
                    f"for properties in this premium segment."
                )
            else:
                return (
                    f"⚠️ **High-End Property Detected**\n\n"
                    f"The estimated value for this property is notably higher than the "
                    f"historical average for {location} ({multiple:.1f}x the typical price).\n\n"
                    f"**Predicted**: ₹{predicted_ppsf:,.0f} per sq.ft\n"
                    f"**Area Average**: ₹{reference_mean:,.0f} per sq.ft\n\n"
                    f"This can occur with:\n"
                    f"• Premium properties with luxury amenities\n"
                    f"• Properties in prime micro-locations\n"
                    f"• Recent market appreciation not reflected in historical data\n\n"
                    f"**Recommendation**: For high-value properties, we recommend "
                    f"a professional appraisal to account for unique features."
                )
        
        elif outlier_type == "too_low":
            fraction = validation_details.get("fraction_of_average", 0)
            
            return (
                f"⚠️ **Below Market Value Detected**\n\n"
                f"The estimated value appears unusually low for {location} "
                f"(only {fraction:.1%} of the area average).\n\n"
                f"**Predicted**: ₹{predicted_ppsf:,.0f} per sq.ft\n"
                f"**Area Average**: ₹{reference_mean:,.0f} per sq.ft\n\n"
                f"This could indicate:\n"
                f"• Property requiring significant renovation\n"
                f"• Distress sale conditions\n"
                f"• Data input errors\n\n"
                f"**Recommendation**: Please verify the property details and "
                f"consult local experts for accurate valuation."
            )
        
        return ""
    
    def get_location_insights(self, location: str) -> Dict[str, Any]:
        """Get market insights for a specific location.
        
        Args:
            location: Property location
            
        Returns:
            Dictionary with location-specific market insights
        """
        if not location:
            return {}
        
        location_clean = location.strip().lower()
        
        # Check for exact or partial match
        stats = None
        if location_clean in self.location_stats:
            stats = self.location_stats[location_clean]
        else:
            # Try partial match
            matching_locations = [
                loc for loc in self.location_stats 
                if location_clean in loc or loc in location_clean
            ]
            if matching_locations:
                stats = self.location_stats[matching_locations[0]]
                location_clean = matching_locations[0]
        
        if stats:
            return {
                "location": location,
                "mean_price_per_sqft": stats['mean_price_per_sqft'],
                "median_price_per_sqft": stats['median_price_per_sqft'],
                "price_range_per_sqft": (stats['min_price_per_sqft'], stats['max_price_per_sqft']),
                "sample_size": stats['sample_size'],
                "market_segment": self._classify_market_segment(stats['median_price_per_sqft'])
            }
        
        return {
            "location": location,
            "message": "Limited historical data available for this location"
        }
    
    def _classify_market_segment(self, price_per_sqft: float) -> str:
        """Classify market segment based on price per sqft.
        
        Args:
            price_per_sqft: Price per square foot
            
        Returns:
            Market segment classification
        """
        if price_per_sqft < 3000:
            return "Budget"
        elif price_per_sqft < 5000:
            return "Mid-Range"
        elif price_per_sqft < 8000:
            return "Premium"
        elif price_per_sqft < 12000:
            return "Luxury"
        else:
            return "Ultra-Luxury"
    
    def save_statistics(self, filepath: Path = None):
        """Save calculated statistics to JSON file.
        
        Args:
            filepath: Path to save statistics
        """
        if filepath is None:
            filepath = Path(__file__).parent.parent / "data" / "location_statistics.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        stats_to_save = {
            "global_stats": self.global_stats,
            "location_stats": self.location_stats,
            "metadata": {
                "total_properties": len(self.data) if hasattr(self, 'data') else 0,
                "total_locations": len(self.location_stats),
                "data_path": str(self.data_path)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        logger.info(f"Statistics saved to {filepath}")
