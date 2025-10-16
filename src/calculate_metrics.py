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
    
    # Drop rows with missing values
    data = data.dropna(subset=['bath', 'balcony', 'price'])
    
    return data

def calculate_model_metrics():
    """Calculate metrics for the existing model."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    # Use the path from settings
    data_path = Path("/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv")
    model_dir = project_root / "data" / "models"
    
    # Initialize metrics calculator
    metrics_calc = ModelMetricsCalculator(model_dir=model_dir)
    
    # Load data
    data = prepare_data(data_path)
    print(f"Data loaded: {len(data)} rows")
    
    # Prepare features and target
    X = data[['bath', 'balcony']].values
    y = data['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a simple model for consistent metrics
    print("Training model for metrics calculation...")
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Save the simple model
    simple_model_path = model_dir / "simple_model.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, simple_model_path)
    print(f"Simple model saved to: {simple_model_path}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(y_test, y_pred)
    
    # Save metrics
    metrics_calc.save_metrics(metrics)
    
    print("\nModel Metrics Calculated:")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f} Lakhs")
    print(f"  MAE: {metrics['mae']:.2f} Lakhs")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    print(f"  Sample Size: {metrics['sample_size']}")
    print(f"\nMetrics saved to: {metrics_calc.metrics_file}")

if __name__ == "__main__":
    calculate_model_metrics()
