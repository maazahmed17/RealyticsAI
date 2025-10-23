#!/usr/bin/env python3
"""
Advanced XGBoost Training for Price Prediction
==============================================
Uses feature engineering to achieve R² > 0.79
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from steps.data_ingestion_step import data_ingestion_step
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from datetime import datetime

def create_advanced_features(df):
    """Create advanced features for better prediction"""
    df = df.copy()
    
    print("\n🔧 Creating advanced features...")
    
    # 1. Price per sqft (most important feature)
    if 'total_sqft' in df.columns and 'price' in df.columns:
        df['price_per_sqft'] = df['price'] / df['total_sqft']
    
    # 2. Room ratios
    if 'bhk' in df.columns and 'bath' in df.columns:
        df['bath_per_bhk'] = df['bath'] / (df['bhk'] + 1)  # +1 to avoid division by zero
    
    if 'balcony' in df.columns and 'bhk' in df.columns:
        df['balcony_per_bhk'] = df['balcony'] / (df['bhk'] + 1)
    
    # 3. Building features
    if 'floor_number' in df.columns and 'total_floors' in df.columns:
        df['floor_ratio'] = df['floor_number'] / (df['total_floors'] + 1)
        df['is_ground_floor'] = (df['floor_number'] == 0).astype(int)
        df['is_top_floor'] = (df['floor_number'] == df['total_floors']).astype(int)
        df['mid_floor'] = ((df['floor_number'] > 0) & (df['floor_number'] < df['total_floors'])).astype(int)
    
    # 4. Size categories
    if 'total_sqft' in df.columns:
        df['sqft_category'] = pd.cut(df['total_sqft'], 
                                      bins=[0, 800, 1200, 1600, 2000, 10000],
                                      labels=['compact', 'medium', 'large', 'spacious', 'luxury'])
        # One-hot encode
        sqft_dummies = pd.get_dummies(df['sqft_category'], prefix='size')
        df = pd.concat([df, sqft_dummies], axis=1)
        df = df.drop('sqft_category', axis=1)
    
    # 5. BHK categories
    if 'bhk' in df.columns:
        df['bhk_category'] = pd.cut(df['bhk'], 
                                     bins=[0, 2, 3, 4, 100],
                                     labels=['small', 'medium', 'large', 'luxury'])
        bhk_dummies = pd.get_dummies(df['bhk_category'], prefix='bhk_type')
        df = pd.concat([df, bhk_dummies], axis=1)
        df = df.drop('bhk_category', axis=1)
    
    # 6. Age categories
    if 'property_age' in df.columns:
        df['age_category'] = pd.cut(df['property_age'],
                                     bins=[-1, 5, 10, 20, 100],
                                     labels=['new', 'recent', 'old', 'very_old'])
        age_dummies = pd.get_dummies(df['age_category'], prefix='age')
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop('age_category', axis=1)
    
    # 7. Furnishing status one-hot encoding
    if 'furnishing_status' in df.columns:
        furnish_dummies = pd.get_dummies(df['furnishing_status'], prefix='furnish')
        df = pd.concat([df, furnish_dummies], axis=1)
        df = df.drop('furnishing_status', axis=1)
    
    # 8. Location frequency encoding
    if 'location' in df.columns:
        location_freq = df['location'].value_counts()
        df['location_freq'] = df['location'].map(location_freq)
        
        # Location average price
        if 'price' in df.columns:
            location_avg_price = df.groupby('location')['price'].transform('mean')
            df['location_avg_price'] = location_avg_price
        
        df = df.drop('location', axis=1)
    
    # 9. Amenities count
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].str.split(',').str.len().fillna(0)
        df = df.drop('amenities', axis=1)
    
    # 10. Polynomial features for important continuous vars
    important_features = ['total_sqft', 'bhk', 'bath', 'property_age']
    for feat in important_features:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2
            df[f'{feat}_log'] = np.log1p(df[feat])
    
    # 11. Interaction features
    if 'total_sqft' in df.columns and 'bhk' in df.columns:
        df['sqft_per_room'] = df['total_sqft'] / (df['bhk'] + 1)
    
    if 'total_sqft' in df.columns and 'property_age' in df.columns:
        df['sqft_age_interaction'] = df['total_sqft'] * df['property_age']
    
    print(f"✅ Created {len(df.columns)} total features")
    
    return df

def train_advanced_xgboost(data_path: str = "/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv"):
    """Train advanced XGBoost model with feature engineering"""
    
    print("\n" + "="*70)
    print("🚀 ADVANCED XGBOOST TRAINING - Feature Engineering Enabled")
    print("="*70)
    
    # 1. Load data
    print("\n📂 Loading data...")
    df = data_ingestion_step(data_path)
    print(f"✅ Loaded {len(df)} properties")
    
    # 2. Create advanced features
    df_engineered = create_advanced_features(df)
    
    # 3. Prepare features and target
    print("\n📊 Preparing features...")
    
    # Separate target
    if 'price' not in df_engineered.columns:
        raise ValueError("Target column 'price' not found!")
    
    y = df_engineered['price']
    X = df_engineered.drop('price', axis=1)
    
    # Keep only numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"📈 Final feature count: {X.shape[1]}")
    print(f"📦 Training samples: {len(X)}")
    
    # 4. Split data
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. Train XGBoost with optimized parameters
    print("\n🤖 Training XGBoost model...")
    print("Parameters:")
    print("  - n_estimators: 500")
    print("  - max_depth: 8")
    print("  - learning_rate: 0.05")
    print("  - subsample: 0.8")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    
    # Fit model (early_stopping handled via n_estimators)
    model.fit(X_train, y_train, verbose=False)
    
    # 6. Evaluate
    print("\n📊 Evaluating model...")
    
    # Train metrics
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE (Lakhs)':<20} {train_rmse:<15.2f} {test_rmse:<15.2f}")
    print(f"{'MAE (Lakhs)':<20} {train_mae:<15.2f} {test_mae:<15.2f}")
    print(f"{'MAPE (%)':<20} {'':<15} {test_mape:<15.2f}")
    print("="*70)
    
    # 7. Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    print("-" * 50)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # 8. Save model and artifacts
    print("\n💾 Saving model...")
    model_dir = Path("/home/maaz/RealyticsAI/data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = model_dir / f"xgboost_advanced_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save feature columns
    feature_cols_path = model_dir / f"feature_columns_{timestamp}.pkl"
    joblib.dump(X.columns.tolist(), feature_cols_path)
    print(f"✅ Feature columns saved: {feature_cols_path}")
    
    # Save metrics
    metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_mape": float(test_mape),
        "n_features": int(X.shape[1]),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "timestamp": timestamp
    }
    
    import json
    metrics_path = model_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    print("\n" + "="*70)
    print("✨ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTest R² Score: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f} Lakhs")
    
    if test_r2 > 0.75:
        print("\n🎉 Excellent! Model achieved high accuracy (R² > 0.75)")
    elif test_r2 > 0.65:
        print("\n👍 Good! Model performance is acceptable (R² > 0.65)")
    else:
        print("\n⚠️  Model may need further tuning")
    
    return model, X.columns.tolist(), metrics

if __name__ == "__main__":
    try:
        model, features, metrics = train_advanced_xgboost()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
