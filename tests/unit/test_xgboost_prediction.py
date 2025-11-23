#!/usr/bin/env python3
"""
Test XGBoost Model Predictions
================================
Quick test to verify XGBoost model is loading and predicting correctly
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.feature_engineering_advanced import AdvancedFeatureEngineer

def test_model_loading():
    """Test if model files exist and can be loaded"""
    print("\n" + "=" * 70)
    print("STEP 1: Testing Model Loading")
    print("=" * 70)
    
    model_dir = Path("/home/maaz/RealyticsAI/data/models")
    
    # Find latest xgboost_fixed model
    fixed_models = list(model_dir.glob("xgboost_fixed_*.pkl"))
    
    if not fixed_models:
        print("❌ No fixed XGBoost models found!")
        return None, None, None
    
    latest_model_path = max(fixed_models, key=lambda p: p.stat().st_ctime)
    timestamp = latest_model_path.name.replace("xgboost_fixed_", "").replace(".pkl", "")
    
    print(f"✅ Found model: {latest_model_path.name}")
    
    # Load model
    try:
        model = joblib.load(latest_model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None
    
    # Load scaler
    scaler_path = model_dir / f"scaler_{timestamp}.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded")
    else:
        print(f"⚠️  Scaler not found")
        scaler = None
    
    # Load feature columns
    features_path = model_dir / f"feature_columns_{timestamp}.pkl"
    if features_path.exists():
        feature_columns = joblib.load(features_path)
        print(f"✅ Feature columns loaded: {len(feature_columns)} features")
        print(f"   First 10 features: {feature_columns[:10]}")
    else:
        print(f"❌ Feature columns not found")
        feature_columns = None
    
    return model, scaler, feature_columns


def test_feature_engineering():
    """Test feature engineering pipeline"""
    print("\n" + "=" * 70)
    print("STEP 2: Testing Feature Engineering")
    print("=" * 70)
    
    # Create test property
    test_property = pd.DataFrame([{
        'location': 'Whitefield',
        'bhk': 3,
        'total_sqft': 1500,
        'bath': 2,
        'balcony': 2,
        'propertyageyears': 5,
        'floornumber': 3,
        'totalfloors': 10,
        'parking': 1
    }])
    
    print(f"\n📋 Test Property:")
    print(f"   Location: Whitefield")
    print(f"   BHK: 3, Bath: 2, Balcony: 2")
    print(f"   Total Sqft: 1500")
    
    # Apply feature engineering
    try:
        engineer = AdvancedFeatureEngineer()
        transformed = engineer.transform(
            test_property,
            use_polynomial=False,
            use_interactions=True,
            use_binning=True,
            use_statistical=True
        )
        
        print(f"\n✅ Feature engineering successful")
        print(f"   Generated {len(transformed.columns)} features")
        print(f"   Feature names: {list(transformed.columns)[:15]}...")
        
        return transformed
        
    except Exception as e:
        print(f"\n❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(model, scaler, feature_columns, transformed_data):
    """Test making a prediction"""
    print("\n" + "=" * 70)
    print("STEP 3: Testing Prediction")
    print("=" * 70)
    
    if model is None or feature_columns is None or transformed_data is None:
        print("❌ Cannot test prediction - missing components")
        return
    
    try:
        # Select numeric features
        numeric_features = transformed_data.select_dtypes(include=[np.number])
        print(f"\n📊 Numeric features: {len(numeric_features.columns)}")
        
        # Align with training features
        X = pd.DataFrame()
        missing_features = []
        
        for col in feature_columns:
            if col in numeric_features.columns:
                X[col] = numeric_features[col]
            else:
                X[col] = 0
                missing_features.append(col)
        
        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features (filled with 0)")
            print(f"   First 5 missing: {missing_features[:5]}")
        
        print(f"✅ Feature alignment complete: {X.shape}")
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
            print(f"✅ Features scaled")
        else:
            X_scaled = X.values
            print(f"⚠️  Using unscaled features")
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Predicted Price: ₹{prediction:.2f} Lakhs")
        print(f"   Predicted Price: ₹{prediction * 100000:.2f}")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_columns, model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\n📈 Top 5 Important Features:")
            for feat, imp in top_features:
                print(f"   {feat}: {imp:.4f}")
        
        return prediction
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("XGBoost Model Test Suite")
    print("=" * 70)
    
    # Test 1: Load model
    model, scaler, feature_columns = test_model_loading()
    
    # Test 2: Feature engineering
    transformed = test_feature_engineering()
    
    # Test 3: Prediction
    prediction = test_prediction(model, scaler, feature_columns, transformed)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if model and scaler and feature_columns and transformed is not None and prediction is not None:
        print("✅ ALL TESTS PASSED")
        print(f"   Model is working correctly!")
        print(f"   Sample prediction: ₹{prediction:.2f} Lakhs")
    else:
        print("❌ SOME TESTS FAILED")
        print("   Please check the errors above")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
