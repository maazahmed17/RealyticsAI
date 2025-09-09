#!/usr/bin/env python3
"""
Test script to verify that the enhanced model gives different prices
for different property specifications
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import RealyticsAIChatbot

def test_price_predictions():
    """Test price predictions for different properties"""
    
    print("Testing Enhanced Price Prediction Model...")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = RealyticsAIChatbot()
    
    # Test properties
    test_properties = [
        {
            'name': 'Whitefield 3BHK',
            'bhk': 3,
            'bath': 2,
            'balcony': 1,
            'sqft': 1500,
            'location': 'whitefield'
        },
        {
            'name': 'Hebbal 3BHK',
            'bhk': 3,
            'bath': 2,
            'balcony': 1,
            'sqft': 1200,
            'location': 'hebbal'
        },
        {
            'name': 'Electronic City 2BHK',
            'bhk': 2,
            'bath': 2,
            'balcony': 1,
            'sqft': 1000,
            'location': 'electronic city'
        },
        {
            'name': 'Koramangala 4BHK',
            'bhk': 4,
            'bath': 3,
            'balcony': 2,
            'sqft': 2000,
            'location': 'koramangala'
        }
    ]
    
    results = []
    
    for prop in test_properties:
        try:
            if (chatbot.price_model and chatbot.feature_columns and 
                chatbot.location_encoder and chatbot.feature_scaler):
                # Use enhanced model
                predicted_price = chatbot._predict_with_enhanced_model(
                    prop['bhk'], prop['bath'], prop['balcony'], 
                    prop['sqft'], prop['location']
                )
                model_type = "Enhanced XGBoost"
            else:
                # Fallback estimation
                predicted_price = 100.0  # Default
                model_type = "Fallback"
                
            results.append({
                'property': prop['name'],
                'price': predicted_price,
                'model': model_type,
                'bhk': prop['bhk'],
                'sqft': prop['sqft'],
                'location': prop['location']
            })
            
        except Exception as e:
            print(f"Error predicting {prop['name']}: {e}")
            results.append({
                'property': prop['name'],
                'price': 'ERROR',
                'model': 'Failed',
                'bhk': prop['bhk'],
                'sqft': prop['sqft'],
                'location': prop['location']
            })
    
    # Print results
    print(f"{'Property':<20} {'Location':<15} {'BHK':<4} {'Sqft':<6} {'Price (Lakhs)':<12} {'Model':<15}")
    print("-" * 80)
    
    for result in results:
        price_str = f"₹{result['price']:.2f}" if isinstance(result['price'], float) else result['price']
        print(f"{result['property']:<20} {result['location']:<15} {result['bhk']:<4} {result['sqft']:<6} {price_str:<12} {result['model']:<15}")
    
    # Check if we got different prices
    prices = [r['price'] for r in results if isinstance(r['price'], float)]
    if len(set(prices)) > 1:
        print("\n✅ SUCCESS: Model gives different prices for different properties!")
        print(f"Price range: ₹{min(prices):.2f} - ₹{max(prices):.2f} Lakhs")
    else:
        print("\n❌ ISSUE: All properties have the same price")
        
    return results

if __name__ == "__main__":
    test_price_predictions()
