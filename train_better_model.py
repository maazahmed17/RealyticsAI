#!/usr/bin/env python3
"""
Retrain Price Prediction Model with Better Location Handling
===========================================================
This script trains a new XGBoost model with proper location encoding
and feature engineering for accurate predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the Bengaluru dataset"""
    print("📂 Loading dataset...")
    
    # Check for data file
    data_path = Path("data/bengaluru_house_prices.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure you have the Bengaluru house prices dataset")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} properties")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Clean column names (handle different formats)
    df.columns = df.columns.str.strip()
    
    # Common column mappings - your dataset already has good column names!
    column_mapping = {
        'total_sqft': 'TotalSqft',
        'bhk': 'BHK', 
        'bath': 'Bath',
        'balcony': 'Balcony',
        'location': 'Location',
        'price': 'PriceLakhs',
        'Price': 'PriceLakhs'  # Your dataset uses 'Price'
    }
    
    # Apply mappings where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Handle total_sqft ranges (e.g., "1000-1200")
    if 'total_sqft' in df.columns:
        def parse_sqft(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val)
            if '-' in val_str:
                try:
                    parts = val_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return np.nan
            try:
                return float(val)
            except:
                return np.nan
        
        df['TotalSqft'] = df['total_sqft'].apply(parse_sqft)
    
    # Extract BHK from size column if needed
    if 'size' in df.columns and 'BHK' not in df.columns:
        df['BHK'] = df['size'].str.extract('(\d+)').astype(float)
    
    # Ensure we have the required columns
    required_cols = ['BHK', 'TotalSqft', 'Bath', 'Balcony', 'Location', 'PriceLakhs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return None
    
    # Select and clean the data
    df_clean = df[required_cols].copy()
    
    # Remove rows with missing critical data
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"🧹 Removed {initial_rows - len(df_clean):,} rows with missing data")
    
    # Remove extreme outliers
    for col in ['TotalSqft', 'PriceLakhs']:
        Q1 = df_clean[col].quantile(0.01)
        Q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= Q1) & (df_clean[col] <= Q3)]
    
    print(f"✅ Final dataset: {len(df_clean):,} properties")
    print(f"📍 Unique locations: {df_clean['Location'].nunique()}")
    print(f"💰 Price range: ₹{df_clean['PriceLakhs'].min():.1f}L - ₹{df_clean['PriceLakhs'].max():.1f}L")
    
    return df_clean

def create_features(df):
    """Create engineered features"""
    print("🔧 Creating features...")
    
    df_features = df.copy()
    
    # Add default features that the model expects
    df_features['PropertyAgeYears'] = 5  # Default age
    df_features['FloorNumber'] = 2      # Default floor
    df_features['TotalFloors'] = 4      # Default total floors  
    df_features['Parking'] = 1          # Default parking
    
    # Location encoding - use frequency encoding to preserve location importance
    location_counts = df_features['Location'].value_counts()
    df_features['LocationFrequency'] = df_features['Location'].map(location_counts)
    
    # Location price encoding (target encoding with regularization)
    location_price = df_features.groupby('Location')['PriceLakhs'].mean()
    global_mean = df_features['PriceLakhs'].mean()
    
    # Smooth location prices with global mean
    location_price_smoothed = {}
    for loc, price in location_price.items():
        count = location_counts[loc]
        # More smoothing for locations with fewer samples
        smoothing = 10
        weight = count / (count + smoothing)
        smoothed_price = weight * price + (1 - weight) * global_mean
        location_price_smoothed[loc] = smoothed_price
    
    df_features['LocationPriceEncoding'] = df_features['Location'].map(location_price_smoothed)
    
    # Create interaction features
    df_features['SqftPerBHK'] = df_features['TotalSqft'] / df_features['BHK']
    df_features['BathPerBHK'] = df_features['Bath'] / df_features['BHK']
    df_features['TotalRooms'] = df_features['BHK'] + df_features['Bath']
    
    print(f"✅ Created features. Total columns: {len(df_features.columns)}")
    return df_features

def train_model(df):
    """Train XGBoost model"""
    print("🤖 Training XGBoost model...")
    
    # Prepare features and target
    feature_cols = ['BHK', 'TotalSqft', 'Bath', 'Balcony', 'PropertyAgeYears', 
                   'FloorNumber', 'TotalFloors', 'Parking', 'LocationFrequency', 
                   'LocationPriceEncoding', 'SqftPerBHK', 'BathPerBHK', 'TotalRooms']
    
    X = df[feature_cols].copy()
    y = df['PriceLakhs'].copy()
    
    print(f"📊 Features: {feature_cols}")
    print(f"🎯 Target: PriceLakhs (range: {y.min():.1f} - {y.max():.1f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Training in progress...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("📈 Model Performance:")
    print(f"   R² Score - Train: {train_r2:.4f}, Test: {test_r2:.4f}")
    print(f"   MAE - Train: {train_mae:.2f}L, Test: {test_mae:.2f}L")
    
    # Save model and components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("data/models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / f"xgboost_fixed_{timestamp}.pkl"
    scaler_path = model_dir / f"scaler_{timestamp}.pkl"
    features_path = model_dir / f"feature_columns_{timestamp}.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    
    print("💾 Model saved:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Features: {features_path}")
    
    # Test with sample predictions
    print("\n🔮 Sample Predictions:")
    test_cases = [
        {'BHK': 1, 'TotalSqft': 600, 'Bath': 1, 'Balcony': 0, 'Location': 'Koramangala'},
        {'BHK': 2, 'TotalSqft': 1200, 'Bath': 2, 'Balcony': 1, 'Location': 'HSR Layout'},
        {'BHK': 3, 'TotalSqft': 1800, 'Bath': 3, 'Balcony': 2, 'Location': 'Whitefield'},
        {'BHK': 4, 'TotalSqft': 2500, 'Bath': 4, 'Balcony': 3, 'Location': 'Indiranagar'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        # Create feature vector
        feature_vector = {
            'BHK': test_case['BHK'],
            'TotalSqft': test_case['TotalSqft'],
            'Bath': test_case['Bath'],
            'Balcony': test_case['Balcony'],
            'PropertyAgeYears': 5,
            'FloorNumber': 2,
            'TotalFloors': 4,
            'Parking': 1,
            'LocationFrequency': df[df['Location'] == test_case['Location']]['LocationFrequency'].iloc[0] if test_case['Location'] in df['Location'].values else 10,
            'LocationPriceEncoding': df[df['Location'] == test_case['Location']]['LocationPriceEncoding'].iloc[0] if test_case['Location'] in df['Location'].values else global_mean,
            'SqftPerBHK': test_case['TotalSqft'] / test_case['BHK'],
            'BathPerBHK': test_case['Bath'] / test_case['BHK'],
            'TotalRooms': test_case['BHK'] + test_case['Bath']
        }
        
        test_df = pd.DataFrame([feature_vector])
        test_scaled = scaler.transform(test_df)
        prediction = model.predict(test_scaled)[0]
        
        print(f"   {i}. {test_case['BHK']}BHK in {test_case['Location']}: ₹{prediction:.2f}L")
    
    return model, scaler, feature_cols

def main():
    """Main training function"""
    print("🏠 RealyticsAI Model Training")
    print("=" * 50)
    
    # Load and clean data
    df = load_and_clean_data()
    if df is None:
        return
    
    # Create features
    df_features = create_features(df)
    
    # Train model
    model, scaler, features = train_model(df_features)
    
    print("\n✅ Training completed successfully!")
    print("🚀 You can now test your predictions with run_unified_system.py")

if __name__ == "__main__":
    main()