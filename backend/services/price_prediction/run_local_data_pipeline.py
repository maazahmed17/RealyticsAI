#!/usr/bin/env python3
"""
Flexible pipeline runner for local data integration.
Supports CSV, Excel, and ZIP files with automatic feature detection.
"""

import click
import pandas as pd
from steps.data_ingestion_step import data_ingestion_step
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def analyze_data(file_path: str):
    """Analyze the local data to understand its structure."""
    print("🔍 ANALYZING YOUR LOCAL DATA")
    print("=" * 50)
    
    # Load the data
    try:
        df = data_ingestion_step(file_path)
        
        print(f"✅ File loaded successfully!")
        print(f"📊 Data Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"📋 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📝 COLUMN ANALYSIS:")
        print("-" * 30)
        
        # Analyze columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"🔢 Numerical columns ({len(numerical_cols)}): {numerical_cols[:10]}")
        if len(numerical_cols) > 10:
            print(f"    ... and {len(numerical_cols) - 10} more")
            
        print(f"📝 Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}")
        if len(categorical_cols) > 10:
            print(f"    ... and {len(categorical_cols) - 10} more")
        
        print("\n💡 POTENTIAL TARGET COLUMNS:")
        print("-" * 30)
        
        # Identify potential price/target columns
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['price', 'cost', 'value', 'amount', 'sale']):
                potential_targets.append(col)
        
        if potential_targets:
            print(f"🎯 Found potential target columns: {potential_targets}")
            for target in potential_targets:
                if df[target].dtype in ['int64', 'float64']:
                    print(f"   - {target}: {df[target].min():,.0f} to {df[target].max():,.0f}")
        else:
            print("⚠️  No obvious price/target columns found.")
            print("   Please identify your target column manually.")
        
        print("\n📊 DATA QUALITY:")
        print("-" * 20)
        missing_data = df.isnull().sum()
        print(f"🔍 Missing values: {missing_data.sum():,} total")
        if missing_data.sum() > 0:
            print("   Columns with missing data:")
            for col, missing in missing_data[missing_data > 0].items():
                print(f"   - {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        print("\n🎯 INTEGRATION RECOMMENDATIONS:")
        print("-" * 35)
        print("1. ✅ Your data is compatible with the system!")
        print("2. 🔧 You may need to:")
        if not potential_targets:
            print("   - Identify your target column (what you want to predict)")
        print("   - Map your columns to expected feature names (optional)")
        print("   - Adjust preprocessing steps if needed")
        
        return df, potential_targets
        
    except Exception as e:
        print(f"❌ Error analyzing data: {str(e)}")
        return None, []


def create_custom_pipeline(file_path: str, target_column: str = None, features_to_transform: list = None):
    """Create a custom pipeline for your local data."""
    
    if features_to_transform is None:
        features_to_transform = []
    
    print(f"\n🚀 CREATING CUSTOM PIPELINE")
    print("=" * 40)
    print(f"📁 Data file: {file_path}")
    print(f"🎯 Target column: {target_column}")
    print(f"🔧 Features to transform: {features_to_transform}")
    
    # Import pipeline components
    from steps.data_splitter_step import data_splitter_step
    from steps.feature_engineering_step import feature_engineering_step
    from steps.handle_missing_values_step import handle_missing_values_step
    from steps.model_building_step import model_building_step
    from steps.model_evaluator_step import model_evaluator_step
    from steps.outlier_detection_step import outlier_detection_step
    from zenml import Model, pipeline
    
    @pipeline(
        model=Model(name="local_prices_predictor"),
    )
    def local_data_pipeline():
        """Custom pipeline for local data."""
        
        # Data Ingestion
        raw_data = data_ingestion_step(file_path=file_path)
        
        # Handle Missing Values
        filled_data = handle_missing_values_step(raw_data, strategy="mean")
        
        # Feature Engineering (if specified)
        if features_to_transform:
            engineered_data = feature_engineering_step(
                filled_data, 
                strategy="log", 
                features=features_to_transform
            )
        else:
            engineered_data = filled_data
        
        # Outlier Detection (if target column specified)
        if target_column:
            clean_data = outlier_detection_step(engineered_data, column_name=target_column)
            
            # Data Splitting
            X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column=target_column)
            
            # Model Building
            model = model_building_step(X_train=X_train, y_train=y_train)
            
            # Model Evaluation
            evaluation_metrics, mse = model_evaluator_step(
                trained_model=model, X_test=X_test, y_test=y_test
            )
            
            return model
        else:
            return engineered_data
    
    return local_data_pipeline


@click.command()
@click.option('--file-path', '-f', required=True, help='Path to your local data file (CSV, Excel, or ZIP)')
@click.option('--target-column', '-t', help='Name of the target column (what you want to predict)')
@click.option('--analyze-only', '-a', is_flag=True, help='Only analyze the data, don\'t train model')
@click.option('--features-to-transform', '-ft', help='Comma-separated list of features to transform')
def main(file_path: str, target_column: str = None, analyze_only: bool = False, features_to_transform: str = None):
    """
    Run the price predictor on your local data.
    
    Examples:
    python run_local_data_pipeline.py -f /path/to/your/data.csv -a
    python run_local_data_pipeline.py -f /path/to/your/data.csv -t SalePrice
    python run_local_data_pipeline.py -f /path/to/your/data.xlsx -t Price -ft "Area,Bedrooms"
    """
    
    print("🏠 LOCAL DATA PRICE PREDICTOR")
    print("=" * 50)
    
    # Parse features to transform
    transform_features = []
    if features_to_transform:
        transform_features = [f.strip() for f in features_to_transform.split(',')]
    
    # Analyze the data first
    df, potential_targets = analyze_data(file_path)
    
    if df is None:
        return
    
    if analyze_only:
        print("\n✅ Analysis complete. Use the insights above to configure your pipeline.")
        return
    
    # Auto-detect target column if not provided
    if not target_column and potential_targets:
        target_column = potential_targets[0]
        print(f"\n🎯 Auto-selected target column: {target_column}")
    
    if not target_column:
        print("\n❌ No target column specified or detected.")
        print("   Use --target-column to specify what you want to predict.")
        return
    
    if target_column not in df.columns:
        print(f"\n❌ Target column '{target_column}' not found in data.")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Create and run custom pipeline
    try:
        pipeline_func = create_custom_pipeline(file_path, target_column, transform_features)
        run = pipeline_func()
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"✅ Model trained on your local data")
        print(f"🎯 Target column: {target_column}")
        print(f"📊 Data shape: {df.shape}")
        
        print(
            "\nTo view experiment tracking, run:\n"
            f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        )
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("💡 Try running with --analyze-only first to check data compatibility.")


if __name__ == "__main__":
    main()
