#!/usr/bin/env python3
"""
Enhanced Price Prediction Training Script
==========================================
This script trains an enhanced price prediction model using:
- Advanced feature engineering
- Multiple model algorithms
- Hyperparameter tuning
- Ensemble methods
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import mlflow
import mlflow.sklearn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

# Import custom modules
from models.feature_engineering_advanced import AdvancedFeatureEngineer
from models.model_building_advanced import (
    ModelBuilder, XGBoostStrategy, LightGBMStrategy, 
    EnhancedRandomForestStrategy, GradientBoostingStrategy,
    EnsembleStrategy, compare_models
)
from models.hyperparameter_tuning import HyperparameterTuner, quick_tune

warnings.filterwarnings('ignore')
console = Console()


class EnhancedPricePredictionTrainer:
    """Enhanced trainer for price prediction models"""
    
    def __init__(self, data_path: str = None):
        """Initialize the trainer"""
        self.data_path = data_path or "/home/maaz/RealyticsAI/data/raw/bengaluru_house_prices.csv"
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_model = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.results = {}
        
        # MLflow setup
        mlflow.set_tracking_uri("file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns")
        mlflow.set_experiment("enhanced_price_prediction")
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        console.print("\n[bold cyan]📂 Loading and Preparing Data[/bold cyan]")
        console.print("=" * 70)
        
        try:
            # Load data
            self.data = pd.read_csv(self.data_path)
            console.print(f"[green]✅ Loaded {len(self.data):,} properties[/green]")
            console.print(f"[cyan]📊 Original shape: {self.data.shape}[/cyan]")
            
            # Basic cleaning
            self._clean_data()
            
            # Display data info
            self._display_data_info()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error loading data: {e}[/red]")
            return False
    
    def _clean_data(self):
        """Basic data cleaning"""
        console.print("\n[yellow]🧹 Cleaning data...[/yellow]")
        
        # Handle 'total_sqft' column if it exists and contains ranges
        if 'total_sqft' in self.data.columns:
            def convert_sqft(x):
                try:
                    if '-' in str(x):
                        parts = str(x).split('-')
                        return (float(parts[0]) + float(parts[1])) / 2
                    return float(x)
                except:
                    return np.nan
            
            self.data['total_sqft'] = self.data['total_sqft'].apply(convert_sqft)
        
        # Handle 'size' column to extract BHK
        if 'size' in self.data.columns and 'bhk' not in self.data.columns:
            self.data['bhk'] = self.data['size'].str.extract('(\d+)').astype(float)
        
        # Drop rows with critical missing values
        critical_cols = ['price', 'total_sqft']
        for col in critical_cols:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        # Remove extreme outliers using IQR method
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in ['price', 'total_sqft', 'bath', 'bhk']:
            if col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        console.print(f"[green]✅ Data cleaned. Final shape: {self.data.shape}[/green]")
    
    def _display_data_info(self):
        """Display data information"""
        table = Table(title="Dataset Information", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Total Properties", f"{len(self.data):,}")
        table.add_row("Features", str(len(self.data.columns)))
        table.add_row("Numeric Features", str(len(self.data.select_dtypes(include=[np.number]).columns)))
        table.add_row("Categorical Features", str(len(self.data.select_dtypes(include=['object']).columns)))
        
        if 'price' in self.data.columns:
            table.add_row("Avg Price", f"₹{self.data['price'].mean():.2f} Lakhs")
            table.add_row("Price Range", f"₹{self.data['price'].min():.1f} - {self.data['price'].max():.1f} Lakhs")
        
        console.print(table)
    
    def engineer_features(self):
        """Apply advanced feature engineering"""
        console.print("\n[bold cyan]🔧 Feature Engineering[/bold cyan]")
        console.print("=" * 70)
        
        # Apply comprehensive feature engineering
        console.print("[yellow]Creating advanced features...[/yellow]")
        
        self.data_engineered = self.feature_engineer.transform(
            self.data,
            use_polynomial=True,
            use_interactions=True,
            use_binning=True,
            use_statistical=True,
            polynomial_degree=2
        )
        
        console.print(f"[green]✅ Features created. Total features: {len(self.data_engineered.columns)}[/green]")
        
        # Prepare features and target
        feature_cols = [col for col in self.data_engineered.columns if col != 'price']
        numeric_features = self.data_engineered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = self.data_engineered[numeric_features]
        y = self.data_engineered['price']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove features with zero variance
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()], index=X.index)
        
        console.print(f"[cyan]📊 Final feature shape: {X.shape}[/cyan]")
        
        # Split data with validation set for early stopping
        # First split: 80% train+val, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        console.print(f"[green]✅ Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}[/green]")
        
        # Feature importance analysis
        self._analyze_feature_importance()
    
    def _analyze_feature_importance(self):
        """Analyze feature importance using mutual information"""
        from sklearn.feature_selection import mutual_info_regression
        
        console.print("\n[yellow]📊 Analyzing feature importance...[/yellow]")
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=42)
        mi_scores = pd.Series(mi_scores, index=self.X_train.columns).sort_values(ascending=False)
        
        # Display top features
        table = Table(title="Top 15 Most Important Features", show_header=True, header_style="bold magenta")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance Score", style="yellow")
        
        for feature, score in mi_scores.head(15).items():
            table.add_row(feature, f"{score:.4f}")
        
        console.print(table)
        
        # Select top features for modeling
        top_n = min(50, len(mi_scores))
        self.selected_features = mi_scores.head(top_n).index.tolist()
        
        # Update training, validation, and test sets with selected features
        self.X_train = self.X_train[self.selected_features]
        self.X_val = self.X_val[self.selected_features]
        self.X_test = self.X_test[self.selected_features]
        
        console.print(f"[green]✅ Selected top {len(self.selected_features)} features for modeling[/green]")
    
    def train_models(self, use_tuning: bool = True):
        """Train multiple models with optional hyperparameter tuning"""
        console.print("\n[bold cyan]🤖 Model Training[/bold cyan]")
        console.print("=" * 70)
        
        models_to_train = [
            ('XGBoost', XGBoostStrategy()),
            ('LightGBM', LightGBMStrategy()),
            ('Random Forest', EnhancedRandomForestStrategy()),
            ('Gradient Boosting', GradientBoostingStrategy()),
            ('Voting Ensemble', EnsembleStrategy(ensemble_type='voting')),
            ('Stacking Ensemble', EnsembleStrategy(ensemble_type='stacking'))
        ]
        
        results = []
        
        for model_name, strategy in track(models_to_train, description="Training models..."):
            console.print(f"\n[yellow]Training {model_name}...[/yellow]")
            
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("n_features", len(self.selected_features))
                mlflow.log_param("train_size", len(self.X_train))
                mlflow.log_param("val_size", len(self.X_val))
                mlflow.log_param("test_size", len(self.X_test))
                
                # Build and train model WITH VALIDATION SET
                builder = ModelBuilder(strategy)
                
                # Pass validation set to XGBoost for early stopping
                if model_name == 'XGBoost':
                    model = builder.build_model(self.X_train, self.y_train, 
                                               X_val=self.X_val, y_val=self.y_val)
                elif use_tuning and model_name in ['XGBoost', 'LightGBM', 'Random Forest']:
                    # Use hyperparameter tuning for key models
                    console.print(f"[cyan]🔍 Tuning hyperparameters for {model_name}...[/cyan]")
                    tuner = HyperparameterTuner(scoring='r2', cv=5)
                    
                    model_type_map = {
                        'XGBoost': 'xgboost',
                        'LightGBM': 'lightgbm',
                        'Random Forest': 'random_forest'
                    }
                    
                    model, best_params = tuner.auto_tune(
                        model_type_map[model_name],
                        self.X_train, self.y_train,
                        method='random'
                    )
                    
                    # Log best parameters
                    for param, value in best_params.items():
                        mlflow.log_param(f"best_{param}", value)
                else:
                    model = builder.build_model(self.X_train, self.y_train)
                
                # Evaluate model
                train_metrics = builder.evaluate_model(model, self.X_train, self.y_train)
                test_metrics = builder.evaluate_model(model, self.X_test, self.y_test)
                
                # Log metrics
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value)
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Store results
                result = {
                    'Model': model_name,
                    'Train R²': train_metrics['r2'],
                    'Test R²': test_metrics['r2'],
                    'Train RMSE': train_metrics['rmse'],
                    'Test RMSE': test_metrics['rmse'],
                    'Train MAE': train_metrics['mae'],
                    'Test MAE': test_metrics['mae'],
                    'model': model
                }
                results.append(result)
                
                console.print(f"[green]✅ {model_name} - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.2f}[/green]")
        
        # Convert results to DataFrame
        self.results = pd.DataFrame(results)
        
        # Find best model
        best_idx = self.results['Test R²'].idxmax()
        self.best_model = self.results.loc[best_idx, 'model']
        best_model_name = self.results.loc[best_idx, 'Model']
        
        console.print(f"\n[bold green]🏆 Best Model: {best_model_name}[/bold green]")
        
        # Display results table
        self._display_results()
    
    def _display_results(self):
        """Display model comparison results"""
        table = Table(title="🎯 Model Performance Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Train R²", justify="right", style="green")
        table.add_column("Test R²", justify="right", style="yellow")
        table.add_column("Train RMSE", justify="right", style="green")
        table.add_column("Test RMSE", justify="right", style="yellow")
        table.add_column("Train MAE", justify="right", style="green")
        table.add_column("Test MAE", justify="right", style="yellow")
        
        # Sort by Test R²
        results_sorted = self.results.sort_values('Test R²', ascending=False)
        
        for _, row in results_sorted.iterrows():
            table.add_row(
                row['Model'],
                f"{row['Train R²']:.4f}",
                f"{row['Test R²']:.4f}",
                f"{row['Train RMSE']:.2f}",
                f"{row['Test RMSE']:.2f}",
                f"{row['Train MAE']:.2f}",
                f"{row['Test MAE']:.2f}"
            )
        
        console.print(table)
        
        # Performance improvement summary
        best_r2 = results_sorted.iloc[0]['Test R²']
        improvement = (best_r2 - 0.1936) / 0.1936 * 100  # Comparing with your original R² of 0.1936
        
        panel = Panel(
            f"""[bold green]Performance Improvement Summary[/bold green]
            
Original Model R²: 0.1936
Enhanced Model R²: {best_r2:.4f}
[bold cyan]Improvement: {improvement:.1f}%[/bold cyan]

The enhanced model explains {best_r2*100:.1f}% of the variance in property prices,
compared to only 19.36% with the original model.""",
            border_style="green"
        )
        console.print(panel)
    
    def save_best_model(self):
        """Save the best model to disk"""
        console.print("\n[bold cyan]💾 Saving Best Model[/bold cyan]")
        console.print("=" * 70)
        
        # Create models directory
        models_dir = Path(__file__).parent.parent.parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = models_dir / f"enhanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(self.best_model, model_path)
        
        # Save feature list
        feature_path = models_dir / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(feature_path, 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Save scaler if needed
        scaler = RobustScaler()
        scaler.fit(self.X_train)
        scaler_path = models_dir / f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(scaler, scaler_path)
        
        console.print(f"[green]✅ Model saved to: {model_path}[/green]")
        console.print(f"[green]✅ Features saved to: {feature_path}[/green]")
        console.print(f"[green]✅ Scaler saved to: {scaler_path}[/green]")
    
    def run_complete_pipeline(self, use_tuning: bool = True):
        """Run the complete training pipeline"""
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]🚀 ENHANCED PRICE PREDICTION MODEL TRAINING[/bold cyan]")
        console.print("=" * 80)
        console.print(f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        
        # Load data
        if not self.load_and_prepare_data():
            return
        
        # Feature engineering
        self.engineer_features()
        
        # Train models
        self.train_models(use_tuning=use_tuning)
        
        # Save best model
        self.save_best_model()
        
        # Final summary
        console.print("\n" + "=" * 80)
        console.print("[bold green]✅ TRAINING COMPLETE![/bold green]")
        console.print("=" * 80)
        
        best_model_info = self.results.loc[self.results['Test R²'].idxmax()]
        
        summary = {
            "Best Model": best_model_info['Model'],
            "Test R² Score": f"{best_model_info['Test R²']:.4f}",
            "Test RMSE": f"₹{best_model_info['Test RMSE']:.2f} Lakhs",
            "Test MAE": f"₹{best_model_info['Test MAE']:.2f} Lakhs",
            "Features Used": len(self.selected_features),
            "Training Samples": len(self.X_train),
            "Test Samples": len(self.X_test)
        }
        
        for key, value in summary.items():
            console.print(f"  • {key}: {value}")
        
        console.print(f"\n[dim]Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Enhanced Price Prediction Model")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedPricePredictionTrainer(data_path=args.data)
    
    # Run pipeline
    trainer.run_complete_pipeline(use_tuning=not args.no_tuning)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
