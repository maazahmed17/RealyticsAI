"""
Professional Prediction Formatter Module
========================================
Formats prediction results with credibility, transparency, and professional presentation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.metrics_calculator import ModelMetricsCalculator
from models.comparables_finder import ComparablesFinder


class PredictionFormatter:
    """Format prediction results professionally with transparency."""
    
    def __init__(self):
        """Initialize formatter with metrics and comparables finder."""
        self.metrics_calculator = ModelMetricsCalculator()
        self.comparables_finder = ComparablesFinder()
    
    def format_prediction_results(self,
                                 prediction: float,
                                 property_features: Dict[str, Any],
                                 model_type: str = "ensemble",
                                 include_market_context: bool = True) -> str:
        """
        Format prediction results with professional presentation.
        
        Args:
            prediction: The predicted price in Lakhs
            property_features: Dictionary containing bhk, sqft, location, bath, balcony
            model_type: Type of model used for prediction
            include_market_context: Whether to include market context
            
        Returns:
            Formatted string with professional prediction presentation
        """
        # Extract features
        bhk = property_features.get('bhk', 2)
        sqft = property_features.get('sqft', 1200)
        location = property_features.get('location', '')
        bath = property_features.get('bath', 2)
        balcony = property_features.get('balcony', 1)
        
        # Find comparable properties with refined logic
        comparables, comp_stats = self.comparables_finder.find_comparables(
            location=location,
            bhk=bhk,
            sqft=sqft,
            strict=True  # Start with strict matching
        )
        
        # Get metrics
        mape = self.metrics_calculator.get_mape()
        mae = self.metrics_calculator.get_mae()
        
        # Calculate defensible price range using MAE
        lower_bound, upper_bound = self.metrics_calculator.get_confidence_range(prediction)
        
        # Build the response
        lines = []
        
        # Header with model confidence (not accuracy claim)
        comp_count = comp_stats.get('count', 0)
        if comp_count > 0:
            lines.append(f"**Property Valuation Analysis**")
            lines.append(f"Based on our analysis of {comp_count} similar properties in the area:")
        else:
            lines.append(f"**Property Valuation Estimate**")
            lines.append(f"Based on market analysis (limited comparable data available):")
        
        lines.append("")
        
        # Property details section
        lines.append("**Property Details:**")
        lines.append(f"• Type: {bhk} BHK")
        if sqft and sqft > 0:
            lines.append(f"• Size: {sqft:.0f} sq.ft")
            lines.append(f"• Price per sq.ft: ₹{(prediction * 100000 / sqft):.0f}")
        lines.append(f"• Bathrooms: {bath}")
        lines.append(f"• Balconies: {balcony}")
        if location:
            lines.append(f"• Location: {location}")
        
        lines.append("")
        
        # Estimated price with confidence range
        lines.append(f"**Estimated Value: ₹{prediction:.2f} Lakhs**")
        lines.append("")
        
        # Defensible price range using MAE
        lines.append(f"**Expected Price Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f} Lakhs")
        
        # Add search criteria transparency
        if comp_count > 0:
            search_mode = comp_stats.get('search_mode', 'strict')
            if search_mode == 'relaxed':
                lines.append(f"*(Using relaxed search criteria due to limited exact matches)*")
        
        lines.append("")
        
        # Market context if comparables found
        if comp_count > 0 and include_market_context:
            lines.append("**Market Context:**")
            
            # Comparable properties stats
            if 'median_price' in comp_stats:
                median = comp_stats['median_price']
                percentile_25 = comp_stats.get('price_25_percentile', median * 0.85)
                percentile_75 = comp_stats.get('price_75_percentile', median * 1.15)
                
                lines.append(f"• Median price in area: ₹{median:.2f} Lakhs")
                lines.append(f"• Market range (25th-75th percentile): ₹{percentile_25:.2f} - ₹{percentile_75:.2f} Lakhs")
                
                # Position relative to market
                if prediction < percentile_25:
                    lines.append("• Position: Below market rate (good value)")
                elif prediction > percentile_75:
                    lines.append("• Position: Above market rate (premium pricing)")
                else:
                    lines.append("• Position: Within typical market range")
            
            lines.append("")
        
        # Model confidence footer with MAPE
        lines.append("**Model Confidence:**")
        lines.append(f"💡 Our model is typically accurate within {mape}% of the final sale price")
        
        # Professional disclaimer
        lines.append("")
        lines.append("*Note: This is an AI-generated estimate based on historical data and market trends. "
                    "Actual prices may vary based on specific property conditions, exact location, "
                    "amenities, and current market dynamics. For precise valuation, please consult "
                    "local real estate professionals.*")
        
        return "\n".join(lines)
    
    def format_error_response(self, error_type: str = "general") -> str:
        """
        Format error responses professionally.
        
        Args:
            error_type: Type of error to format
            
        Returns:
            Formatted error message
        """
        if error_type == "insufficient_data":
            return ("I apologize, but I don't have enough information to provide an accurate estimate. "
                   "Please provide:\n"
                   "• Number of bedrooms (BHK)\n"
                   "• Location\n"
                   "• Size in square feet (optional)\n"
                   "• Number of bathrooms (optional)")
        
        elif error_type == "invalid_location":
            return ("I couldn't find sufficient data for the specified location. "
                   "Please check the spelling or try a nearby area. "
                   "Popular locations include Whitefield, Koramangala, Electronic City, etc.")
        
        elif error_type == "model_unavailable":
            return ("I apologize, but the prediction model is temporarily unavailable. "
                   "Please try again in a few moments.")
        
        else:
            return ("I encountered an unexpected issue while processing your request. "
                   "Please try rephrasing your query or contact support if the issue persists.")
    
    def format_brief_response(self, prediction: float, confidence_range: tuple) -> str:
        """
        Format a brief response for quick interactions.
        
        Args:
            prediction: The predicted price
            confidence_range: Tuple of (lower, upper) bounds
            
        Returns:
            Brief formatted response
        """
        lower, upper = confidence_range
        mape = self.metrics_calculator.get_mape()
        
        return (f"**Estimated Price:** ₹{prediction:.2f} Lakhs\n"
                f"**Range:** ₹{lower:.2f} - ₹{upper:.2f} Lakhs\n"
                f"*Typically accurate within {mape}% of actual price*")
