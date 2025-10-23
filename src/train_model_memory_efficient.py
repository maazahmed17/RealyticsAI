#!/usr/bin/env python3
"""
Memory-Efficient Training Script
=================================
Trains model without OOM errors by:
1. Tuning on a small sample (30k-50k rows)
2. Training final model on full dataset with best parameters
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import gc

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import xgboost as xgb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.feature_engineering_advanced import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')

print("=" * 80)
print("MEMORY-EFFICIENT PRICE PREDICTION MODEL TRAINING")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def load_and_clean_data(data_path, sample_size=None):
    """Load and clean data with optional sampling"""
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Basic cleaning
    print("\n🧹 Cleaning data...")
    
    # Handle 'total_sqft' ranges
    if 'total_sqft' in df.columns:
        def convert_sqft(x):
            try:
                if '-' in str(x):
                    parts = str(x).split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                return float(x)
            except:
                return np.nan
        df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    
    # Extract BHK from size
    if 'size' in df.columns and 'bhk' not in df.columns:
        df['bhk'] = df['size'].str.extract('(\d+)').astype(float)
    
    # Drop critical NaNs
    critical_cols = ['price', 'total_sqft']
    for col in critical_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    # Remove extreme outliers (3 IQR method)
    for col in ['price', 'total_sqft', 'bath', 'bhk']:
        if col in df.columns and col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    print(f"✅ After cleaning: {len(df):,} rows")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        print(f"\n🎲 Sampling {sample_size:,} rows for hyperparameter tuning...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled dataset: {len(df):,} rows")
    
    return df


def engineer_features(df, feature_engineer=None, fit=True):
    """Apply feature engineering"""
    print("\n🔧 Engineering features...")
    
    if feature_engineer is None:
        feature_engineer = AdvancedFeatureEngineer()
    
    # Apply transformations (without polynomial to save memory)
    df_transformed = feature_engineer.transform(
        df,
        use_polynomial=False,  # Skip polynomial to save memory
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
    X_filtered = selector.fit_transform(X) if fit else selector.transform(X)
    selected_cols = X.columns[selector.get_support()]
    X = pd.DataFrame(X_filtered, columns=selected_cols, index=X.index)
    
    print(f"✅ Final feature shape: {X.shape}")
    
    return X, y, feature_engineer, selected_cols.tolist()


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Quick hyperparameter tuning with RandomizedSearchCV"""
    print("\n🔍 Tuning hyperparameters on sample...")
    print(f"Training samples: {len(X_train):,}")
    
    # Define parameter grid
    param_distributions = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.3, 0.5, 0.7],
        'reg_lambda': [0.8, 1.0, 1.2],
        'min_child_weight': [2, 3, 4],
        'n_estimators': [300, 500]
    }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    
    # Randomized search (faster than GridSearchCV)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=20,  # Try 20 combinations
        scoring='r2',
        cv=3,  # 3-fold CV for speed
        verbose=1,
        n_jobs=2,  # Use 2 cores to avoid OOM
        random_state=42
    )
    
    # Fit
    print("⏳ Running randomized search (this may take 5-10 minutes)...")
    random_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\n✅ Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Evaluate on validation set
    best_model = random_search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"\n📊 Validation Performance:")
    print(f"   R² Score: {val_r2:.4f}")
    print(f"   RMSE: {val_rmse:.2f} Lakhs")
    
    return random_search.best_params_


def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    """Train final model with best parameters"""
    print("\n🤖 Training final model on full dataset with best parameters...")
    print(f"Training samples: {len(X_train):,}")
    
    # Create model with best parameters
    model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with early stopping
    print("⏳ Training with early stopping...")
    try:
        # Try newer XGBoost API (v2.0+)
        from xgboost.callback import EarlyStopping
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[EarlyStopping(rounds=50, save_best=True)],
            verbose=False
        )
    except:
        # Fallback to older API
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
    
    print(f"✅ Training stopped at {model.best_iteration} iterations")
    
    # Evaluate
    print("\n📊 Final Model Performance:")
    
    # Train metrics
    train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    
    # Test metrics
    test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE (Lakhs)':<20} {train_rmse:<15.2f} {test_rmse:<15.2f}")
    print(f"{'MAE (Lakhs)':<20} {train_mae:<15.2f} {test_mae:<15.2f}")
    
    # Check for overfitting
    gap_ratio = train_rmse / test_rmse if test_rmse > 0 else 0
    print(f"\n📈 Train/Test RMSE Ratio: {gap_ratio:.2f}x")
    
    if gap_ratio < 1.3:
        print("   ✅ Healthy gap - no overfitting detected!")
    elif gap_ratio < 1.5:
        print("   ⚠️  Slight overfitting - acceptable")
    else:
        print("   ❌ Significant overfitting detected!")
    
    if test_r2 < 0.6:
        print("   ⚠️  Warning: Low R² score. Model may need more features or data.")
    elif test_r2 >= 0.85:
        print("   ✅ Excellent R² score!")
    
    return model, scaler


def save_model(model, scaler, feature_columns, best_params):
    """Save model, scaler, and metadata"""
    print("\n💾 Saving model...")
    
    models_dir = Path(__file__).parent.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = models_dir / f"xgboost_fixed_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / f"scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")
    
    # Save feature columns
    features_path = models_dir / f"feature_columns_{timestamp}.pkl"
    joblib.dump(feature_columns, features_path)
    print(f"✅ Features saved: {features_path}")
    
    # Save best parameters
    params_path = models_dir / f"best_params_{timestamp}.txt"
    with open(params_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write("=" * 50 + "\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    print(f"✅ Parameters saved: {params_path}")


def main():
    """Main training pipeline"""
    
    # Paths
    data_path = Path("/home/maaz/RealyticsAI/data/raw/bengaluru_house_prices.csv")
    
    # Step 1: Load full data and check size
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    df_full = pd.read_csv(data_path)
    full_size = len(df_full)
    print(f"Full dataset size: {full_size:,} rows")
    del df_full
    gc.collect()
    
    # Determine if we need to sample for tuning
    TUNE_SAMPLE_SIZE = 40000  # Use 40k rows for tuning
    use_sampling = full_size > 50000
    
    if use_sampling:
        print(f"\n📊 Large dataset detected. Will use 2-stage training:")
        print(f"   Stage 1: Tune on {TUNE_SAMPLE_SIZE:,} samples")
        print(f"   Stage 2: Train on full {full_size:,} dataset")
    
    # Step 2: Tune hyperparameters on sample
    print("\n" + "=" * 80)
    print("STEP 2: HYPERPARAMETER TUNING")
    print("=" * 80)
    
    if use_sampling:
        # Load sample for tuning
        df_sample = load_and_clean_data(data_path, sample_size=TUNE_SAMPLE_SIZE)
        
        # Engineer features
        X_sample, y_sample, feature_engineer, feature_cols = engineer_features(df_sample, fit=True)
        
        # Split for tuning
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42
        )
        
        # Tune
        best_params = tune_hyperparameters(X_train_tune, y_train_tune, X_val_tune, y_val_tune)
        
        # Free memory
        del df_sample, X_sample, y_sample, X_train_tune, X_val_tune, y_train_tune, y_val_tune
        gc.collect()
    else:
        # Small dataset - use default good parameters
        print("Small dataset - using optimized default parameters")
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'n_estimators': 1000
        }
    
    # Step 3: Train final model on full dataset
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 80)
    
    # Load full data
    df_full = load_and_clean_data(data_path, sample_size=None)
    
    # Engineer features
    X_full, y_full, feature_engineer, feature_cols = engineer_features(df_full, fit=True)
    
    # Split data (60% train, 20% val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"✅ Split complete:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X_full)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X_full)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X_full)*100:.1f}%)")
    
    # Train final model
    model, scaler = train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params)
    
    # Save model
    save_model(model, scaler, feature_cols, best_params)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel ready for production use!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
