"""
Metrics Calculator Module
=========================
Calculates and stores model evaluation metrics including MAPE and MAE
for use in prediction formatting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import json
from typing import Dict, Any, Optional


class ModelMetricsCalculator:
    """Calculate and store model evaluation metrics."""
    
    def __init__(self, model_dir: Path = None):
        """Initialize metrics calculator.
        
        Args:
            model_dir: Directory to store/load metrics
        """
        self.model_dir = model_dir or Path(__file__).parent.parent.parent / "data" / "models"
        self.metrics_file = self.model_dir / "model_metrics.json"
        self.metrics = self.load_metrics()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing all metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Calculate median absolute error for robustness
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Calculate confidence intervals (using MAE)
        confidence_95_width = 1.96 * mae  # Approximate 95% confidence interval
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2_score": float(r2),
            "median_absolute_error": float(median_ae),
            "confidence_95_width": float(confidence_95_width),
            "sample_size": len(y_true)
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Dictionary of metrics to save
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        import datetime
        metrics["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.metrics = metrics
    
    def load_metrics(self) -> Optional[Dict[str, float]]:
        """Load metrics from file.
        
        Returns:
            Dictionary of metrics or None if file doesn't exist
        """
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_metric(self, metric_name: str, default: float = None) -> float:
        """Get a specific metric value.
        
        Args:
            metric_name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Metric value or default
        """
        if self.metrics:
            return self.metrics.get(metric_name, default)
        return default
    
    def get_mape(self) -> float:
        """Get MAPE value formatted to 1 decimal place."""
        mape = self.get_metric("mape", 10.0)  # Default to 10% if not found
        return round(mape, 1)
    
    def get_mae(self) -> float:
        """Get MAE value in Lakhs."""
        mae = self.get_metric("mae", 5.0)  # Default to 5 Lakhs if not found
        return round(mae, 2)
    
    def get_confidence_range(self, prediction: float) -> tuple:
        """Get confidence range for a prediction.
        
        Args:
            prediction: The predicted value
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mae = self.get_mae()
        lower = max(0, prediction - mae)  # Ensure non-negative
        upper = prediction + mae
        return (round(lower, 2), round(upper, 2))
