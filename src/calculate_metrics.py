#!/usr/bin/env python3
"""
Calculate and Store Model Metrics
==================================
Script to calculate MAPE, MAE, and other metrics for existing models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.metrics_calculator import ModelMetricsCalculator
import sys

def prepare_data(data_path):
    """Load and prepare the data."""
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()

    # Map equivalent columns if necessary
    column_mapping = {
        'pricelakhs': 'price'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in data.columns:
            data[new_col] = data[old_col]

    # Check for required columns and handle missing ones
    required_columns = ['bath', 'balcony', 'price']
    for col in required_columns:
        if col not in data.columns:
            print(f"Warning: Column '{col}' is missing. Filling with default values.")
            if col == 'bath' or col == 'balcony':
                data[col] = 1  # Default to 1 if missing
            elif col == 'price':
                raise ValueError("The 'price' column is mandatory and cannot be missing.")

    # Drop rows with NaN values in critical columns
    data = data.dropna(subset=required_columns)
    
    # Clean total_sqft
    if 'total_sqft' in data.columns:
        def convert_sqft(x):
            try:
                if '-' in str(x):
                    parts = str(x).split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                return float(x)
            except:
                return np.nan
        data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
    
    # Extract BHK
    if 'size' in data.columns and 'bhk' not in data.columns:
        data['bhk'] = data['size'].str.extract('(\d+)').astype(float)
    
    return data

def calculate_model_metrics():
    """Calculate metrics for the XGBoost advanced model."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = Path("/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv")
    model_dir = project_root / "data" / "models"
    
    # Initialize metrics calculator
    metrics_calc = ModelMetricsCalculator(model_dir=model_dir)
    
    print("\n" + "="*70)
    print("CALCULATING METRICS FOR XGBOOST ADVANCED MODEL")
    print("="*70)
    
    # Find the XGBoost advanced model
    xgb_models = list(model_dir.glob("xgboost_advanced_*.pkl"))
    if not xgb_models:
        print("\n❌ No XGBoost advanced model found!")
        print("Please train the model first:")
        print("  cd /home/maaz/RealyticsAI/backend/services/price_prediction")
        print("  python train_xgboost_advanced.py")
        return
    
    # Load the latest XGBoost model
    latest_model_path = max(xgb_models, key=lambda p: p.stat().st_ctime)
    print(f"\n📂 Loading model: {latest_model_path.name}")
    model = joblib.load(latest_model_path)
    
    # Load corresponding feature columns
    timestamp = latest_model_path.name.replace("xgboost_advanced_", "").replace(".pkl", "")
    feature_cols_path = model_dir / f"feature_columns_{timestamp}.pkl"
    
    if not feature_cols_path.exists():
        print(f"\n❌ Feature columns file not found: {feature_cols_path.name}")
        return
    
    feature_columns = joblib.load(feature_cols_path)
    print(f"✅ Loaded {len(feature_columns)} feature columns")
    
    # Load and prepare data using the same preprocessing as training
    print("\n📊 Loading and preparing data...")
    sys.path.insert(0, str(project_root / "backend" / "services" / "price_prediction"))
    from train_xgboost_advanced import create_advanced_features
    from steps.data_ingestion_step import data_ingestion_step
    
    # Load data
    df = data_ingestion_step(str(data_path))
    print(f"✅ Loaded {len(df)} properties")
    
    # Apply same feature engineering
    df_engineered = create_advanced_features(df)
    
    # Prepare features
    if 'price' not in df_engineered.columns:
        raise ValueError("Target column 'price' not found!")
    
    y = df_engineered['price']
    X = df_engineered.drop('price', axis=1)
    
    # Keep only numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Handle missing/infinite values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Ensure we have the same features as the model expects
    X = X[feature_columns]
    
    print(f"📈 Final feature count: {X.shape[1]}")
    print(f"📦 Total samples: {len(X)}")
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✂️  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    train_metrics = metrics_calc.calculate_metrics(y_train, y_pred_train)
    test_metrics = metrics_calc.calculate_metrics(y_test, y_pred_test)
    
    # Combine metrics
    combined_metrics = {
        'model_name': latest_model_path.name,
        'model_type': 'XGBoost Advanced with Feature Engineering',
        'n_features': int(X.shape[1]),
        'n_train_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'train_r2': float(train_metrics['r2_score']),
        'test_r2': float(test_metrics['r2_score']),
        'train_rmse': float(train_metrics['rmse']),
        'test_rmse': float(test_metrics['rmse']),
        'train_mae': float(train_metrics['mae']),
        'test_mae': float(test_metrics['mae']),
        'train_mape': float(train_metrics['mape']),
        'test_mape': float(test_metrics['mape']),
        'timestamp': timestamp
    }
    
    # Save metrics
    metrics_calc.save_metrics(combined_metrics)
    
    # Display results
    print("\n" + "="*70)
    print("📈 MODEL METRICS CALCULATED")
    print("="*70)
    print(f"\nModel: {latest_model_path.name}")
    print(f"Type: XGBoost with {X.shape[1]} engineered features")
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_metrics['r2_score']:<15.4f} {test_metrics['r2_score']:<15.4f}")
    print(f"{'RMSE (Lakhs)':<20} {train_metrics['rmse']:<15.2f} {test_metrics['rmse']:<15.2f}")
    print(f"{'MAE (Lakhs)':<20} {train_metrics['mae']:<15.2f} {test_metrics['mae']:<15.2f}")
    print(f"{'MAPE (%)':<20} {train_metrics['mape']:<15.1f} {test_metrics['mape']:<15.1f}")
    print("="*70)
    print(f"\n✅ Metrics saved to: {metrics_calc.metrics_file}")
    print("\n🎉 The model shows excellent performance!" if test_metrics['r2_score'] > 0.90 else "")

if __name__ == "__main__":
    calculate_model_metrics()
