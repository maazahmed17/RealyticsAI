#!/usr/bin/env python3
"""
Evaluate the Fixed Model
=========================
Calculates metrics for the newly trained xgboost_fixed model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import sys

sys.path.append(str(Path(__file__).parent))
from models.feature_engineering_advanced import AdvancedFeatureEngineer

print("=" * 80)
print("EVALUATING FIXED MODEL (No Data Leakage)")
print("=" * 80)

# Paths
model_dir = Path("/home/maaz/RealyticsAI/data/models")
data_path = Path("/home/maaz/RealyticsAI/data/raw/bengaluru_house_prices.csv")

# Find the latest xgboost_fixed model
print("\n📂 Looking for fixed models...")
fixed_models = list(model_dir.glob("xgboost_fixed_*.pkl"))

if not fixed_models:
    print("\n❌ No fixed model found!")
    print("Please train the model first:")
    print("  cd /home/maaz/RealyticsAI/src")
    print("  python train_model_memory_efficient.py")
    sys.exit(1)

# Load latest model
latest_model_path = max(fixed_models, key=lambda p: p.stat().st_ctime)
timestamp = latest_model_path.name.replace("xgboost_fixed_", "").replace(".pkl", "")

print(f"✅ Found model: {latest_model_path.name}")
print(f"   Created: {latest_model_path.stat().st_mtime}")

# Load model and related files
print("\n📦 Loading model components...")
model = joblib.load(latest_model_path)
print(f"✅ Model loaded")

# Load scaler
scaler_path = model_dir / f"scaler_{timestamp}.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded")
else:
    print("⚠️  No scaler found, will create new one")
    scaler = RobustScaler()

# Load feature columns
features_path = model_dir / f"feature_columns_{timestamp}.pkl"
if features_path.exists():
    feature_columns = joblib.load(features_path)
    print(f"✅ Feature columns loaded ({len(feature_columns)} features)")
else:
    print("❌ Feature columns file not found!")
    sys.exit(1)

# Load and prepare data
print("\n📊 Loading and preparing data...")
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df):,} rows")

# Normalize column names
df.columns = df.columns.str.lower()

# Clean data
def convert_sqft(x):
    try:
        if '-' in str(x):
            parts = str(x).split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return np.nan

if 'totalsqft' in df.columns:
    df['totalsqft'] = df['totalsqft'].apply(convert_sqft)
    df['total_sqft'] = df['totalsqft']  # Create alias
elif 'total_sqft' in df.columns:
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

if 'size' in df.columns and 'bhk' not in df.columns:
    df['bhk'] = df['size'].str.extract('(\d+)').astype(float)

# Drop NaNs
sqft_col = 'totalsqft' if 'totalsqft' in df.columns else 'total_sqft'
df = df.dropna(subset=['price', sqft_col])

# Remove outliers
for col in ['price', 'total_sqft', 'bath', 'bhk']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"✅ After cleaning: {len(df):,} rows")

# Feature engineering
print("\n🔧 Engineering features...")
feature_engineer = AdvancedFeatureEngineer()
df_transformed = feature_engineer.transform(
    df,
    use_polynomial=False,
    use_interactions=True,
    use_binning=True,
    use_statistical=True
)

# Prepare features
feature_cols = [col for col in df_transformed.columns if col != 'price']
numeric_features = df_transformed[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X = df_transformed[numeric_features]
y = df_transformed['price']

# Handle missing values
X = X.fillna(X.median())

# Remove zero variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
selected_cols = X.columns[selector.get_support()]
X = pd.DataFrame(X_filtered, columns=selected_cols, index=X.index)

print(f"✅ Feature shape: {X.shape}")

# Ensure we have the same features
try:
    X = X[feature_columns]
    print(f"✅ Aligned features: {len(feature_columns)}")
except KeyError as e:
    print(f"⚠️  Warning: Some expected features missing: {e}")
    # Use only common features
    common_features = [f for f in feature_columns if f in X.columns]
    print(f"   Using {len(common_features)} common features")
    X = X[common_features]

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✂️  Train: {len(X_train):,} | Test: {len(X_test):,}")

# Scale if scaler was fitted
if hasattr(scaler, 'scale_'):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Make predictions
print("\n🔮 Making predictions...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
print("\n📊 Calculating metrics...")
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

# Display results
print("\n" + "=" * 80)
print("📈 FIXED MODEL EVALUATION RESULTS")
print("=" * 80)
print(f"\nModel: {latest_model_path.name}")
print(f"Features: {len(feature_columns)}")
print(f"Timestamp: {timestamp}")

print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
print("-" * 65)
print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f} {abs(train_r2-test_r2):<15.4f}")
print(f"{'RMSE (Lakhs)':<20} {train_rmse:<15.2f} {test_rmse:<15.2f} {abs(train_rmse-test_rmse):<15.2f}")
print(f"{'MAE (Lakhs)':<20} {train_mae:<15.2f} {test_mae:<15.2f} {abs(train_mae-test_mae):<15.2f}")

# Overfitting check
gap_ratio = train_rmse / test_rmse if test_rmse > 0 else 0
print(f"\n📈 Train/Test RMSE Ratio: {gap_ratio:.2f}x")

if gap_ratio < 1.3:
    print("   ✅ Healthy - NO overfitting detected!")
elif gap_ratio < 1.5:
    print("   ⚠️  Slight overfitting - acceptable")
else:
    print("   ❌ Significant overfitting!")

if test_r2 >= 0.80:
    print(f"\n✅ Excellent R² score ({test_r2:.4f})!")
elif test_r2 >= 0.70:
    print(f"\n👍 Good R² score ({test_r2:.4f})")
else:
    print(f"\n⚠️  R² score could be improved ({test_r2:.4f})")

print("\n" + "=" * 80)

# Save metrics
metrics = {
    'model_name': latest_model_path.name,
    'timestamp': timestamp,
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'gap_ratio': float(gap_ratio),
    'n_features': len(feature_columns),
    'n_train': len(X_train),
    'n_test': len(X_test)
}

import json
metrics_file = model_dir / "fixed_model_metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n💾 Metrics saved to: {metrics_file}")
print("\n✅ Evaluation complete!")
