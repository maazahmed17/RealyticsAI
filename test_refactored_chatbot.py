#!/usr/bin/env python3
"""
Test Script for Refactored Chatbot
===================================
Tests the new professional formatter and refactored chatbot.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from presentation.formatter import PredictionFormatter
from models.comparables_finder import ComparablesFinder

def test_formatter():
    """Test the new professional formatter."""
    
    print("="*60)
    print("Testing Professional Prediction Formatter")
    print("="*60)
    
    # Initialize formatter
    formatter = PredictionFormatter()
    
    # Test case 1: Property with location
    print("\n1. Testing 3 BHK in Whitefield:")
    print("-" * 40)
    
    property_features = {
        'bhk': 3,
        'sqft': 1500,
        'location': 'Whitefield',
        'bath': 2,
        'balcony': 2
    }
    
    result = formatter.format_prediction_results(
        prediction=125.5,
        property_features=property_features,
        model_type="ensemble",
        include_market_context=True
    )
    
    print(result)
    
    # Test case 2: Property without exact location match
    print("\n\n2. Testing 2 BHK in Unknown Area:")
    print("-" * 40)
    
    property_features = {
        'bhk': 2,
        'sqft': 1000,
        'location': 'Some New Area',
        'bath': 2,
        'balcony': 1
    }
    
    result = formatter.format_prediction_results(
        prediction=85.0,
        property_features=property_features,
        model_type="baseline",
        include_market_context=True
    )
    
    print(result)
    
    # Test case 3: Brief response
    print("\n\n3. Testing Brief Response Format:")
    print("-" * 40)
    
    result = formatter.format_brief_response(
        prediction=95.5,
        confidence_range=(85.0, 106.0)
    )
    
    print(result)
    
    # Test case 4: Error response
    print("\n\n4. Testing Error Response:")
    print("-" * 40)
    
    result = formatter.format_error_response("insufficient_data")
    print(result)

def test_comparables():
    """Test the comparables finder."""
    
    print("\n\n" + "="*60)
    print("Testing Comparables Finder")
    print("="*60)
    
    finder = ComparablesFinder()
    
    # Test finding comparables
    print("\n5. Finding comparables for 3 BHK in Whitefield (1500 sqft):")
    print("-" * 40)
    
    comparables, stats = finder.find_comparables(
        location="Whitefield",
        bhk=3,
        sqft=1500,
        strict=True
    )
    
    print(f"Found {stats['count']} comparable properties")
    print(f"Search criteria: {stats['criteria']}")
    print(f"Search mode: {stats['search_mode']}")
    
    if stats['count'] > 0:
        print(f"Average price: ₹{stats.get('avg_price', 0):.2f} Lakhs")
        print(f"Median price: ₹{stats.get('median_price', 0):.2f} Lakhs")
        print(f"Price range (25th-75th): ₹{stats.get('price_25_percentile', 0):.2f} - ₹{stats.get('price_75_percentile', 0):.2f} Lakhs")

if __name__ == "__main__":
    test_formatter()
    test_comparables()
    
    print("\n\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
