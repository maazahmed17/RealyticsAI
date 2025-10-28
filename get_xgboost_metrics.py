#!/usr/bin/env python3
"""
Get XGBoost Model Metrics
=========================
Calculate and display metrics for the XGBoost model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    print("🔍 XGBoost Model Metrics")
    print("=" * 70)
    
    # Find the latest model
    model_dir = Path("data/models")
    xgb_models = list(model_dir.glob("xgboost_fixed_*.pkl"))
    
    if not xgb_models:
        print("❌ No XGBoost model found!")
        print("Please train the model first: python train_better_model.py")
        return
    
    # Load latest model
    latest_model = max(xgb_models, key=lambda p: p.stat().st_ctime)
    timestamp = latest_model.stem.replace("xgboost_fixed_", "")
    
    print(f"\n📂 Loading model: {latest_model.name}")
    model = joblib.load(latest_model)
    
    # Load scaler and features
    scaler_path = model_dir / f"scaler_{timestamp}.pkl"
    features_path = model_dir / f"feature_columns_{timestamp}.pkl"
    
    if not scaler_path.exists() or not features_path.exists():
        print("❌ Scaler or features file not found!")
        return
    
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    print(f"✅ Loaded {len(feature_cols)} features")
    
    # Load and prepare data (same as train_better_model.py)
    data_path = Path("data/bengaluru_house_prices.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} properties")
    
    # Clean data (simplified version of train_better_model.py logic)
    df.columns = df.columns.str.strip()
    
    # Column mappings
    if 'Price' in df.columns:
        df['PriceLakhs'] = df['Price']
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
    
    if 'size' in df.columns and 'BHK' not in df.columns:
        df['BHK'] = df['size'].str.extract('(\\d+)').astype(float)
    
    # Rename columns
    for old, new in [('bhk', 'BHK'), ('bath', 'Bath'), ('balcony', 'Balcony'), 
                     ('location', 'Location')]:
        if old in df.columns:
            df[new] = df[old]
    
    # Select required columns
    required_cols = ['BHK', 'TotalSqft', 'Bath', 'Balcony', 'Location', 'PriceLakhs']
    df_clean = df[required_cols].dropna()
    
    # Remove outliers
    for col in ['TotalSqft', 'PriceLakhs']:
        Q1 = df_clean[col].quantile(0.01)
        Q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= Q1) & (df_clean[col] <= Q3)]
    
    # Create features
    df_clean['PropertyAgeYears'] = 5
    df_clean['FloorNumber'] = 2
    df_clean['TotalFloors'] = 4
    df_clean['Parking'] = 1
    
    location_counts = df_clean['Location'].value_counts()
    df_clean['LocationFrequency'] = df_clean['Location'].map(location_counts)
    
    location_price = df_clean.groupby('Location')['PriceLakhs'].mean()
    global_mean = df_clean['PriceLakhs'].mean()
    
    location_price_smoothed = {}
    for loc, price in location_price.items():
        count = location_counts[loc]
        smoothing = 10
        weight = count / (count + smoothing)
        smoothed_price = weight * price + (1 - weight) * global_mean
        location_price_smoothed[loc] = smoothed_price
    
    df_clean['LocationPriceEncoding'] = df_clean['Location'].map(location_price_smoothed)
    df_clean['SqftPerBHK'] = df_clean['TotalSqft'] / df_clean['BHK']
    df_clean['BathPerBHK'] = df_clean['Bath'] / df_clean['BHK']
    df_clean['TotalRooms'] = df_clean['BHK'] + df_clean['Bath']
    
    # Prepare X and y
    X = df_clean[feature_cols].copy()
    y = df_clean['PriceLakhs'].copy()
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mape = calculate_mape(y_train, y_train_pred)
    test_mape = calculate_mape(y_test, y_test_pred)
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 XGBoost Model Performance")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE (Lakhs)':<20} {train_rmse:<15.2f} {test_rmse:<15.2f}")
    print(f"{'MAE (Lakhs)':<20} {train_mae:<15.2f} {test_mae:<15.2f}")
    print(f"{'MAPE (%)':<20} {train_mape:<15.2f} {test_mape:<15.2f}")
    print("-" * 50)
    print(f"{'Training samples':<20} {len(X_train):<15}")
    print(f"{'Test samples':<20} {len(X_test):<15}")
    print(f"{'Features':<20} {len(feature_cols):<15}")
    print("=" * 70)
    
    # Save metrics to JSON
    import json
    metrics = {
        "model_name": latest_model.name,
        "model_type": "XGBoost",
        "timestamp": timestamp,
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_mape": float(train_mape),
        "test_mape": float(test_mape),
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test)
    }
    
    metrics_path = model_dir / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
