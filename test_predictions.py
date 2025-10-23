#!/usr/bin/env python3
"""
Test Price Predictions
======================
Test that the model gives different predictions for different properties
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.services.price_prediction.fixed_price_predictor import get_price_predictor

print("=" * 80)
print("TESTING PRICE PREDICTIONS - Different Properties Should Get Different Prices")
print("=" * 80)

# Get the predictor
predictor = get_price_predictor()

# Test properties with different characteristics
test_properties = [
    {
        "name": "Small 2BHK in Electronic City",
        "location": "Electronic City",
        "bhk": 2,
        "bath": 2,
        "balcony": 1,
        "totalsqft": 1000
    },
    {
        "name": "Large 3BHK in Whitefield",
        "location": "Whitefield",
        "bhk": 3,
        "bath": 2,
        "balcony": 2,
        "totalsqft": 1700
    },
    {
        "name": "Luxury 4BHK in Koramangala",
        "location": "Koramangala",
        "bhk": 4,
        "bath": 3,
        "balcony": 3,
        "totalsqft": 2500
    },
    {
        "name": "Budget 2BHK in Hebbal",
        "location": "Hebbal",
        "bhk": 2,
        "bath": 1,
        "balcony": 1,
        "totalsqft": 900
    },
    {
        "name": "Premium 3BHK in Indiranagar",
        "location": "Indiranagar",
        "bhk": 3,
        "bath": 3,
        "balcony": 2,
        "totalsqft": 1800
    }
]

print("\nTesting predictions for 5 different properties...\n")

predictions = []
for prop in test_properties:
    result = predictor.predict(prop)
    
    if result['success']:
        price = result['price']
        predictions.append(price)
        
        print(f"📍 {prop['name']}")
        print(f"   Details: {prop['bhk']}BHK, {prop['totalsqft']} sqft, {prop['bath']} bath")
        print(f"   💰 Predicted Price: {result['price_formatted']}")
        print(f"   📊 Confidence: {result['confidence']:.0%}")
        print()
    else:
        print(f"❌ {prop['name']}: ERROR - {result.get('error', 'Unknown error')}\n")

# Check if predictions are different
print("=" * 80)
print("ANALYSIS")
print("=" * 80)

unique_predictions = len(set(predictions))
print(f"\n✅ Total predictions: {len(predictions)}")
print(f"✅ Unique predictions: {unique_predictions}")

if unique_predictions == 1:
    print("\n❌ **PROBLEM DETECTED**: All properties getting the SAME price!")
    print(f"   All predictions are: ₹{predictions[0]:.2f} Lakhs")
    print("\n🔧 This means the feature engineering is not working properly.")
elif unique_predictions == len(predictions):
    print("\n✅ **PERFECT**: All properties getting DIFFERENT prices!")
    print("   The model is working correctly with proper feature engineering.")
else:
    print(f"\n⚠️  **WARNING**: Some properties getting same prices.")
    print(f"   {unique_predictions} unique prices out of {len(predictions)} properties.")

# Show price range
if predictions:
    print(f"\n📊 Price Range:")
    print(f"   Minimum: ₹{min(predictions):.2f} Lakhs")
    print(f"   Maximum: ₹{max(predictions):.2f} Lakhs")
    print(f"   Average: ₹{sum(predictions)/len(predictions):.2f} Lakhs")

print("\n" + "=" * 80)
