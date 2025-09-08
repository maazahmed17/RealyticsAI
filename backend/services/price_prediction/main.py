#!/usr/bin/env python3
"""
RealyticsAI Price Prediction Service - Main Entry Point
========================================================
A comprehensive real estate price prediction system with MLflow tracking,
market analysis, and intelligent insights.

Usage:
    python main.py                    # Run complete analysis with output
    python main.py --mode api         # Start as API server
    python main.py --predict          # Interactive prediction mode
    python main.py --train            # Train new model
"""

import os
import sys
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize Rich console for beautiful output
console = Console()

# Configuration
class Config:
    """Centralized configuration for price prediction"""
    # Data paths
    BENGALURU_DATA_PATH = "/mnt/c/Users/Ahmed/Downloads/bengaluru_house_prices.csv"
    MODEL_SAVE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "models"
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns"
    MLFLOW_EXPERIMENT = "realyticsai_price_prediction"
    MLFLOW_MODEL_NAME = "bengaluru_price_predictor"
    
    # Model configuration
    MODEL_FEATURES = ["bath", "balcony"]
    EXTENDED_FEATURES = ["bath", "balcony", "total_sqft", "bhk"]
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Display settings
    CURRENCY = "₹"
    CURRENCY_UNIT = "Lakhs"
    
    # Analysis settings
    TOP_LOCATIONS = 10
    PRICE_BINS = 5
    
    # Model selection
    USE_ADVANCED_MODEL = False  # Set to True for RandomForest

class PricePredictionSystem:
    """Complete price prediction system with MLflow integration"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_columns = Config.MODEL_FEATURES
        self.is_trained = False
        
        # Setup MLflow
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT)
        
        console.print("[bold green]🚀 RealyticsAI Price Prediction System Initialized[/bold green]")
        
    def load_data(self) -> bool:
        """Load the Bengaluru housing dataset"""
        try:
            console.print("\n[yellow]📂 Loading dataset...[/yellow]")
            
            if not os.path.exists(Config.BENGALURU_DATA_PATH):
                console.print(f"[red]❌ Data file not found: {Config.BENGALURU_DATA_PATH}[/red]")
                return False
            
            self.data = pd.read_csv(Config.BENGALURU_DATA_PATH)
            
            # Basic data info
            console.print(f"[green]✅ Successfully loaded {len(self.data):,} properties[/green]")
            console.print(f"[cyan]📊 Dataset shape: {self.data.shape}[/cyan]")
            console.print(f"[cyan]🏘️ Unique locations: {self.data['location'].nunique()}[/cyan]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error loading data: {e}[/red]")
            return False
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for model training"""
        console.print("\n[yellow]🔧 Preprocessing data...[/yellow]")
        
        # Handle missing values
        df = self.data.copy()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'price':
                df[col] = df[col].fillna(df[col].median())
        
        # Prepare features and target
        if Config.USE_ADVANCED_MODEL and all(col in df.columns for col in Config.EXTENDED_FEATURES):
            self.feature_columns = Config.EXTENDED_FEATURES
        else:
            self.feature_columns = Config.MODEL_FEATURES
        
        X = df[self.feature_columns].copy()
        y = df['price']
        
        # Fill any remaining NaN values
        X = X.fillna(X.mean())
        
        console.print(f"[green]✅ Preprocessed {len(X)} samples with {len(self.feature_columns)} features[/green]")
        
        return X, y
    
    def train_model(self, model_type: str = "linear") -> bool:
        """Train the price prediction model with MLflow tracking"""
        try:
            console.print(f"\n[yellow]🤖 Training {model_type} model with MLflow tracking...[/yellow]")
            
            # Preprocess data
            X, y = self.preprocess_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"bengaluru_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("features", self.feature_columns)
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                mlflow.log_param("total_properties", len(self.data))
                
                # Select and train model
                if model_type == "random_forest":
                    self.model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        random_state=Config.RANDOM_STATE
                    )
                elif model_type == "gradient_boost":
                    self.model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=Config.RANDOM_STATE
                    )
                else:
                    self.model = LinearRegression()
                
                # Train with progress tracking
                with console.status("[bold green]Training model...") as status:
                    self.model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = self.model.predict(X_train)
                test_pred = self.model.predict(X_test)
                
                # Calculate metrics
                self.metrics = {
                    "train_r2": r2_score(y_train, train_pred),
                    "test_r2": r2_score(y_test, test_pred),
                    "train_mse": mean_squared_error(y_train, train_pred),
                    "test_mse": mean_squared_error(y_test, test_pred),
                    "train_mae": mean_absolute_error(y_train, train_pred),
                    "test_mae": mean_absolute_error(y_test, test_pred),
                    "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred))
                }
                
                # Log metrics to MLflow
                for metric_name, value in self.metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log model
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name=Config.MLFLOW_MODEL_NAME
                )
                
                # Save model locally
                Config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
                model_path = Config.MODEL_SAVE_PATH / f"model_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
                joblib.dump(self.model, model_path)
                
                self.is_trained = True
                
                # Display metrics
                self._display_training_metrics()
                
                console.print(f"[green]✅ Model saved to: {model_path}[/green]")
                console.print(f"[green]✅ MLflow run completed successfully![/green]")
                
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Error training model: {e}[/red]")
            return False
    
    def _display_training_metrics(self):
        """Display training metrics in a beautiful table"""
        table = Table(title="🎯 Model Performance Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Training", justify="right", style="green")
        table.add_column("Testing", justify="right", style="yellow")
        
        table.add_row("R² Score", f"{self.metrics['train_r2']:.4f}", f"{self.metrics['test_r2']:.4f}")
        table.add_row("MSE", f"{self.metrics['train_mse']:.2f}", f"{self.metrics['test_mse']:.2f}")
        table.add_row("RMSE", f"{self.metrics['train_rmse']:.2f}", f"{self.metrics['test_rmse']:.2f}")
        table.add_row("MAE", f"{self.metrics['train_mae']:.2f}", f"{self.metrics['test_mae']:.2f}")
        
        console.print(table)
    
    def predict_price(self, features: Dict[str, float], use_hybrid: bool = True) -> Dict[str, Any]:
        """Predict price for given features"""
        if not self.is_trained:
            console.print("[red]❌ Model not trained. Please train first.[/red]")
            return {}
        
        # Prepare input
        X_pred = pd.DataFrame([features])[self.feature_columns]
        
        # ML prediction
        ml_price = self.model.predict(X_pred)[0]
        
        result = {
            "ml_prediction": ml_price,
            "currency": f"{Config.CURRENCY}{ml_price:.2f} {Config.CURRENCY_UNIT}"
        }
        
        if use_hybrid and self.data is not None:
            # Find similar properties
            similar = self.data[
                (self.data['bath'] == features.get('bath', 2)) & 
                (self.data['balcony'] == features.get('balcony', 1))
            ]
            
            if len(similar) > 0:
                avg_price = similar['price'].mean()
                median_price = similar['price'].median()
                
                # Hybrid prediction (60% ML, 40% historical average)
                hybrid_price = 0.6 * ml_price + 0.4 * avg_price
                
                result.update({
                    "similar_properties_count": len(similar),
                    "similar_avg_price": avg_price,
                    "similar_median_price": median_price,
                    "hybrid_prediction": hybrid_price,
                    "hybrid_currency": f"{Config.CURRENCY}{hybrid_price:.2f} {Config.CURRENCY_UNIT}",
                    "price_range": {
                        "min": similar['price'].min(),
                        "max": similar['price'].max()
                    }
                })
        
        return result
    
    def market_analysis(self):
        """Perform comprehensive market analysis"""
        console.print("\n[bold cyan]📊 COMPREHENSIVE MARKET ANALYSIS[/bold cyan]")
        console.print("=" * 70)
        
        # 1. Overall Market Statistics
        self._display_overall_stats()
        
        # 2. Price Distribution Analysis
        self._display_price_distribution()
        
        # 3. Location Analysis
        self._display_location_analysis()
        
        # 4. Feature-based Analysis
        self._display_feature_analysis()
        
        # 5. Correlation Analysis
        self._display_correlation_analysis()
    
    def _display_overall_stats(self):
        """Display overall market statistics"""
        stats = {
            "Total Properties": f"{len(self.data):,}",
            "Average Price": f"{Config.CURRENCY}{self.data['price'].mean():.2f} {Config.CURRENCY_UNIT}",
            "Median Price": f"{Config.CURRENCY}{self.data['price'].median():.2f} {Config.CURRENCY_UNIT}",
            "Price Range": f"{Config.CURRENCY}{self.data['price'].min():.1f} - {self.data['price'].max():.1f} {Config.CURRENCY_UNIT}",
            "Std Deviation": f"{self.data['price'].std():.2f}",
            "Unique Locations": f"{self.data['location'].nunique():,}"
        }
        
        table = Table(title="📈 Overall Market Statistics", show_header=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow")
        
        for key, value in stats.items():
            table.add_row(key, value)
        
        console.print(table)
    
    def _display_price_distribution(self):
        """Display price distribution analysis"""
        console.print("\n[bold]💰 Price Distribution[/bold]")
        
        # Create price bins
        price_bins = pd.qcut(self.data['price'], q=Config.PRICE_BINS)
        distribution = self.data.groupby(price_bins)['price'].agg(['count', 'mean', 'min', 'max'])
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Price Range", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Avg Price", justify="right")
        
        for idx, row in distribution.iterrows():
            range_str = f"{Config.CURRENCY}{row['min']:.1f}-{row['max']:.1f}L"
            table.add_row(
                range_str,
                str(int(row['count'])),
                f"{Config.CURRENCY}{row['mean']:.2f}L"
            )
        
        console.print(table)
    
    def _display_location_analysis(self):
        """Display top locations analysis"""
        console.print(f"\n[bold]🏘️ Top {Config.TOP_LOCATIONS} Locations by Property Count[/bold]")
        
        top_locations = self.data['location'].value_counts().head(Config.TOP_LOCATIONS)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Location", style="cyan", no_wrap=False, width=30)
        table.add_column("Properties", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Price Range", justify="right")
        
        for location, count in top_locations.items():
            loc_data = self.data[self.data['location'] == location]['price']
            table.add_row(
                location[:30],
                str(count),
                f"{Config.CURRENCY}{loc_data.mean():.2f}L",
                f"{Config.CURRENCY}{loc_data.min():.1f}-{loc_data.max():.1f}L"
            )
        
        console.print(table)
    
    def _display_feature_analysis(self):
        """Display analysis by property features"""
        console.print("\n[bold]🏠 Price Analysis by Features[/bold]")
        
        # Bathrooms analysis
        bath_analysis = self.data.groupby('bath')['price'].agg(['mean', 'count', 'std'])
        
        table = Table(title="Bathrooms vs Price", show_header=True, header_style="bold magenta")
        table.add_column("Bathrooms", style="cyan")
        table.add_column("Avg Price", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("Std Dev", justify="right")
        
        for bath, row in bath_analysis.head(6).iterrows():
            if not pd.isna(bath):
                table.add_row(
                    str(int(bath)),
                    f"{Config.CURRENCY}{row['mean']:.2f}L",
                    str(int(row['count'])),
                    f"{row['std']:.2f}"
                )
        
        console.print(table)
        
        # Balcony analysis
        if 'balcony' in self.data.columns:
            balcony_analysis = self.data.groupby('balcony')['price'].agg(['mean', 'count'])
            
            table2 = Table(title="Balconies vs Price", show_header=True, header_style="bold magenta")
            table2.add_column("Balconies", style="cyan")
            table2.add_column("Avg Price", justify="right")
            table2.add_column("Count", justify="right")
            
            for balcony, row in balcony_analysis.head(5).iterrows():
                if not pd.isna(balcony):
                    table2.add_row(
                        str(int(balcony)),
                        f"{Config.CURRENCY}{row['mean']:.2f}L",
                        str(int(row['count']))
                    )
            
            console.print(table2)
    
    def _display_correlation_analysis(self):
        """Display correlation analysis"""
        console.print("\n[bold]🔗 Feature Correlations with Price[/bold]")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != 'price' and not self.data[col].isna().all():
                corr = self.data[col].corr(self.data['price'])
                if not pd.isna(corr):
                    correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Feature", style="cyan")
        table.add_column("Correlation", justify="right")
        table.add_column("Strength", justify="center")
        
        for feature, corr in sorted_corr[:10]:
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            color = "green" if corr > 0 else "red"
            table.add_row(
                feature,
                f"[{color}]{corr:.4f}[/{color}]",
                strength
            )
        
        console.print(table)
    
    def run_predictions_demo(self):
        """Run sample predictions demonstration"""
        console.print("\n[bold cyan]🔮 SAMPLE PREDICTIONS[/bold cyan]")
        console.print("=" * 70)
        
        test_cases = [
            {"bath": 1, "balcony": 0, "desc": "💰 Budget: 1 Bath, No Balcony"},
            {"bath": 2, "balcony": 1, "desc": "🏠 Standard: 2 Bath, 1 Balcony"},
            {"bath": 3, "balcony": 2, "desc": "✨ Premium: 3 Bath, 2 Balconies"},
            {"bath": 4, "balcony": 3, "desc": "🏰 Luxury: 4 Bath, 3 Balconies"},
            {"bath": 5, "balcony": 2, "desc": "👑 Ultra-Luxury: 5 Bath, 2 Balconies"}
        ]
        
        for test in test_cases:
            console.print(f"\n[bold]{test['desc']}[/bold]")
            
            prediction = self.predict_price({
                "bath": test["bath"],
                "balcony": test["balcony"]
            })
            
            if prediction:
                # Create a nice display panel
                content = f"""
🤖 ML Prediction: {Config.CURRENCY}{prediction['ml_prediction']:.2f} {Config.CURRENCY_UNIT}
"""
                
                if 'hybrid_prediction' in prediction:
                    content += f"""🏘️ Similar Properties: {prediction['similar_properties_count']} found
📊 Historical Average: {Config.CURRENCY}{prediction['similar_avg_price']:.2f} {Config.CURRENCY_UNIT}
⭐ Hybrid Prediction: {Config.CURRENCY}{prediction['hybrid_prediction']:.2f} {Config.CURRENCY_UNIT}
📈 Price Range: {Config.CURRENCY}{prediction['price_range']['min']:.1f} - {prediction['price_range']['max']:.1f} {Config.CURRENCY_UNIT}"""
                
                panel = Panel(content.strip(), border_style="green")
                console.print(panel)
    
    def generate_insights(self):
        """Generate and display market insights"""
        console.print("\n[bold cyan]💡 MARKET INSIGHTS & RECOMMENDATIONS[/bold cyan]")
        console.print("=" * 70)
        
        insights = []
        
        # Price trend insight
        avg_price = self.data['price'].mean()
        median_price = self.data['price'].median()
        
        if avg_price > median_price * 1.2:
            insights.append("📈 The market shows signs of luxury properties pulling up average prices")
        elif avg_price < median_price * 0.9:
            insights.append("📉 The market is dominated by budget-friendly properties")
        else:
            insights.append("⚖️ The market shows balanced price distribution")
        
        # Location insights
        top_location = self.data['location'].value_counts().index[0]
        top_loc_price = self.data[self.data['location'] == top_location]['price'].mean()
        
        if top_loc_price > avg_price * 1.2:
            insights.append(f"🏆 {top_location} is a premium location with above-average prices")
        else:
            insights.append(f"💰 {top_location} offers good value with competitive prices")
        
        # Feature insights
        bath_corr = self.data['bath'].corr(self.data['price'])
        if bath_corr > 0.5:
            insights.append("🚿 Number of bathrooms strongly influences property prices")
        
        # Model performance insight
        if self.is_trained and self.metrics:
            if self.metrics['test_r2'] > 0.7:
                insights.append(f"✅ Model shows high accuracy (R²={self.metrics['test_r2']:.3f}) for price predictions")
            elif self.metrics['test_r2'] > 0.5:
                insights.append(f"📊 Model shows moderate accuracy (R²={self.metrics['test_r2']:.3f}) - consider more features")
            else:
                insights.append(f"⚠️ Model accuracy is low (R²={self.metrics['test_r2']:.3f}) - additional data needed")
        
        # Price volatility
        price_std = self.data['price'].std()
        if price_std > avg_price * 0.5:
            insights.append("📊 High price volatility indicates diverse property market")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            console.print(f"  {i}. {insight}")
        
        # Investment recommendations
        console.print("\n[bold]🎯 Investment Recommendations:[/bold]")
        
        # Find undervalued locations
        location_stats = self.data.groupby('location')['price'].agg(['mean', 'count'])
        location_stats = location_stats[location_stats['count'] >= 10]
        undervalued = location_stats.nsmallest(5, 'mean')
        
        console.print("  [green]Potential Value Locations:[/green]")
        for location, stats in undervalued.iterrows():
            console.print(f"    • {location}: Avg {Config.CURRENCY}{stats['mean']:.2f}L ({int(stats['count'])} properties)")
    
    def run_complete_analysis(self):
        """Run complete analysis and display all results"""
        # Header
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]🏠 REALYTICSAI PRICE PREDICTION SYSTEM - COMPLETE ANALYSIS[/bold cyan]")
        console.print("=" * 80)
        console.print(f"[dim]Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        
        # Load data
        if not self.load_data():
            return
        
        # Train model
        if not self.train_model("linear"):
            return
        
        # Market analysis
        self.market_analysis()
        
        # Sample predictions
        self.run_predictions_demo()
        
        # Generate insights
        self.generate_insights()
        
        # MLflow information
        console.print("\n[bold cyan]🔬 MLFLOW TRACKING INFORMATION[/bold cyan]")
        console.print("=" * 70)
        console.print(f"[green]✅ Experiment tracked in MLflow[/green]")
        console.print(f"📁 Tracking URI: {Config.MLFLOW_TRACKING_URI}")
        console.print(f"📊 Experiment Name: {Config.MLFLOW_EXPERIMENT}")
        console.print("\n[yellow]To view MLflow UI, run:[/yellow]")
        console.print(f"[dim]mlflow ui --backend-store-uri {Config.MLFLOW_TRACKING_URI}[/dim]")
        console.print("[dim]Then open: http://localhost:5000[/dim]")
        
        # Summary
        console.print("\n" + "=" * 80)
        console.print("[bold green]✅ ANALYSIS COMPLETE![/bold green]")
        console.print("=" * 80)
        
        summary = {
            "Dataset": f"{len(self.data):,} properties from Bengaluru",
            "Model Performance": f"R² Score = {self.metrics.get('test_r2', 0):.4f}",
            "Average Price": f"{Config.CURRENCY}{self.data['price'].mean():.2f} {Config.CURRENCY_UNIT}",
            "Price Range": f"{Config.CURRENCY}{self.data['price'].min():.1f} - {self.data['price'].max():.1f} {Config.CURRENCY_UNIT}",
            "Top Location": self.data['location'].value_counts().index[0]
        }
        
        for key, value in summary.items():
            console.print(f"  • {key}: {value}")

def start_api_server():
    """Start the FastAPI server"""
    console.print("[bold green]🚀 Starting API Server...[/bold green]")
    console.print("[yellow]API will be available at: http://localhost:8000[/yellow]")
    console.print("[yellow]API Documentation: http://localhost:8000/docs[/yellow]")
    
    # Import and run the FastAPI app
    import uvicorn
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from main_mlflow import app
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

def interactive_prediction(system: PricePredictionSystem):
    """Interactive prediction mode"""
    console.print("\n[bold cyan]🔮 INTERACTIVE PREDICTION MODE[/bold cyan]")
    console.print("Enter 'quit' to exit\n")
    
    while True:
        try:
            console.print("[yellow]Enter property features:[/yellow]")
            
            bath_input = input("Number of bathrooms (default=2): ").strip()
            if bath_input.lower() == 'quit':
                break
            bath = int(bath_input) if bath_input else 2
            
            balcony_input = input("Number of balconies (default=1): ").strip()
            if balcony_input.lower() == 'quit':
                break
            balcony = int(balcony_input) if balcony_input else 1
            
            # Make prediction
            prediction = system.predict_price({"bath": bath, "balcony": balcony})
            
            if prediction:
                console.print("\n[bold green]📊 Prediction Results:[/bold green]")
                console.print(f"ML Prediction: {Config.CURRENCY}{prediction['ml_prediction']:.2f} {Config.CURRENCY_UNIT}")
                
                if 'hybrid_prediction' in prediction:
                    console.print(f"Hybrid Prediction: {Config.CURRENCY}{prediction['hybrid_prediction']:.2f} {Config.CURRENCY_UNIT}")
                    console.print(f"Similar Properties: {prediction['similar_properties_count']}")
                    console.print(f"Price Range: {Config.CURRENCY}{prediction['price_range']['min']:.1f} - {prediction['price_range']['max']:.1f} {Config.CURRENCY_UNIT}")
            
            console.print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[green]Thank you for using RealyticsAI![/green]")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RealyticsAI Price Prediction System")
    parser.add_argument("--mode", choices=["analysis", "api", "predict", "train"], 
                       default="analysis",
                       help="Mode to run: analysis (default), api, predict, or train")
    parser.add_argument("--model", choices=["linear", "random_forest", "gradient_boost"],
                       default="linear",
                       help="Model type to use")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        start_api_server()
    else:
        # Initialize system
        system = PricePredictionSystem()
        
        if args.mode == "train":
            # Just train the model
            if system.load_data():
                system.train_model(args.model)
        elif args.mode == "predict":
            # Interactive prediction mode
            if system.load_data():
                system.train_model(args.model)
                interactive_prediction(system)
        else:
            # Default: Run complete analysis
            system.run_complete_analysis()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)
