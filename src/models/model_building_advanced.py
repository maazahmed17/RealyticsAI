"""
Advanced Model Building Module for Price Prediction
====================================================
This module implements multiple advanced regression algorithms and ensemble methods
to significantly improve the price prediction performance.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    VotingRegressor,
    StackingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import callback
import lightgbm as lgb
from catboost import CatBoostRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelBuildingStrategy(ABC):
    """Abstract base class for model building strategies"""
    
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_val: Optional[pd.DataFrame] = None, 
                            y_val: Optional[pd.Series] = None) -> RegressorMixin:
        """Build and train a model"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model"""
        pass


class XGBoostStrategy(ModelBuildingStrategy):
    """XGBoost Regressor with optimized hyperparameters"""
    
    def __init__(self, **kwargs):
        # Reduced overfitting with stronger regularization
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 1000),
            'max_depth': kwargs.get('max_depth', 5),  # Reduced from 8 to prevent overfitting
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.7),  # Reduced from 0.8
            'gamma': kwargs.get('gamma', 0.1),
            'reg_alpha': kwargs.get('reg_alpha', 0.5),  # Increased L1 regularization
            'reg_lambda': kwargs.get('reg_lambda', 1.0),  # L2 regularization
            'min_child_weight': kwargs.get('min_child_weight', 3),
            'random_state': kwargs.get('random_state', 42),
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'eval_metric': 'rmse'  # Added eval_metric for early stopping
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building XGBoost model with anti-overfitting parameters...")
        
        model = xgb.XGBRegressor(**self.params, verbosity=0)
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        # CRITICAL: Use early stopping with validation set to prevent overfitting
        if X_val is not None and y_val is not None:
            logger.info("Training with early stopping on validation set...")
            # Scale the data first
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Fit with early stopping using callbacks
            eval_set = [(X_val_scaled, y_val)]
            model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                callbacks=[xgb.callback.EarlyStopping(rounds=50)],
                verbose=False
            )
            
            # Create pipeline with fitted components
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
            logger.info(f"Training stopped at {model.best_iteration} iterations (early stopping)")
        else:
            logger.warning("No validation set provided - training without early stopping")
            pipeline.fit(X_train, y_train)
        
        logger.info("XGBoost model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "XGBoost"


class LightGBMStrategy(ModelBuildingStrategy):
    """LightGBM Regressor with optimized hyperparameters"""
    
    def __init__(self, **kwargs):
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 300),
            'max_depth': kwargs.get('max_depth', -1),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'num_leaves': kwargs.get('num_leaves', 50),
            'feature_fraction': kwargs.get('feature_fraction', 0.8),
            'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
            'bagging_freq': kwargs.get('bagging_freq', 5),
            'lambda_l1': kwargs.get('lambda_l1', 0.1),
            'lambda_l2': kwargs.get('lambda_l2', 0.1),
            'min_child_samples': kwargs.get('min_child_samples', 20),
            'random_state': kwargs.get('random_state', 42),
            'verbosity': -1
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building LightGBM model...")
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', lgb.LGBMRegressor(**self.params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("LightGBM model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "LightGBM"


class CatBoostStrategy(ModelBuildingStrategy):
    """CatBoost Regressor with automatic categorical feature handling"""
    
    def __init__(self, cat_features: Optional[List[str]] = None, **kwargs):
        self.cat_features = cat_features or []
        self.params = {
            'iterations': kwargs.get('iterations', 300),
            'depth': kwargs.get('depth', 8),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'l2_leaf_reg': kwargs.get('l2_leaf_reg', 3.0),
            'bagging_temperature': kwargs.get('bagging_temperature', 1),
            'random_strength': kwargs.get('random_strength', 1),
            'border_count': kwargs.get('border_count', 254),
            'random_state': kwargs.get('random_state', 42),
            'verbose': False
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building CatBoost model...")
        
        # CatBoost handles scaling internally
        model = CatBoostRegressor(**self.params)
        
        if self.cat_features:
            cat_feature_indices = [X_train.columns.get_loc(col) for col in self.cat_features 
                                  if col in X_train.columns]
            model.fit(X_train, y_train, cat_features=cat_feature_indices)
        else:
            model.fit(X_train, y_train)
        
        logger.info("CatBoost model training completed")
        return model
    
    def get_model_name(self) -> str:
        return "CatBoost"


class EnhancedRandomForestStrategy(ModelBuildingStrategy):
    """Enhanced Random Forest with optimized parameters"""
    
    def __init__(self, **kwargs):
        # Reduced overfitting with proper constraints
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 200),
            'max_depth': kwargs.get('max_depth', 10),  # Reduced from 20 to prevent overfitting
            'min_samples_split': kwargs.get('min_samples_split', 10),  # Increased from 5
            'min_samples_leaf': kwargs.get('min_samples_leaf', 4),  # Increased from 2
            'max_features': kwargs.get('max_features', 'sqrt'),
            'bootstrap': kwargs.get('bootstrap', True),
            'oob_score': kwargs.get('oob_score', True),
            'random_state': kwargs.get('random_state', 42),
            'n_jobs': -1
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building Enhanced Random Forest model...")
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(**self.params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("Random Forest model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "Enhanced Random Forest"


class GradientBoostingStrategy(ModelBuildingStrategy):
    """Gradient Boosting with optimized parameters"""
    
    def __init__(self, **kwargs):
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 200),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'subsample': kwargs.get('subsample', 0.8),
            'min_samples_split': kwargs.get('min_samples_split', 5),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 3),
            'max_features': kwargs.get('max_features', 'sqrt'),
            'loss': kwargs.get('loss', 'huber'),
            'alpha': kwargs.get('alpha', 0.9),
            'random_state': kwargs.get('random_state', 42)
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building Gradient Boosting model...")
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', GradientBoostingRegressor(**self.params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("Gradient Boosting model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "Gradient Boosting"


class ElasticNetStrategy(ModelBuildingStrategy):
    """ElasticNet Regression with L1 and L2 regularization"""
    
    def __init__(self, **kwargs):
        self.params = {
            'alpha': kwargs.get('alpha', 0.1),
            'l1_ratio': kwargs.get('l1_ratio', 0.5),
            'max_iter': kwargs.get('max_iter', 1000),
            'random_state': kwargs.get('random_state', 42)
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building ElasticNet model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(**self.params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("ElasticNet model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "ElasticNet"


class NeuralNetworkStrategy(ModelBuildingStrategy):
    """Multi-layer Perceptron Neural Network"""
    
    def __init__(self, **kwargs):
        self.params = {
            'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', (100, 50, 25)),
            'activation': kwargs.get('activation', 'relu'),
            'solver': kwargs.get('solver', 'adam'),
            'alpha': kwargs.get('alpha', 0.001),
            'learning_rate': kwargs.get('learning_rate', 'adaptive'),
            'max_iter': kwargs.get('max_iter', 500),
            'early_stopping': kwargs.get('early_stopping', True),
            'validation_fraction': kwargs.get('validation_fraction', 0.1),
            'random_state': kwargs.get('random_state', 42)
        }
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building Neural Network model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(**self.params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("Neural Network model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "Neural Network"


class EnsembleStrategy(ModelBuildingStrategy):
    """Ensemble of multiple models using voting or stacking"""
    
    def __init__(self, ensemble_type: str = 'voting', **kwargs):
        self.ensemble_type = ensemble_type
        self.base_models = []
        
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> RegressorMixin:
        logger.info(f"Building {self.ensemble_type} ensemble model...")
        
        # Create base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)),
            ('lgb', lgb.LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42, verbosity=-1))
        ]
        
        if self.ensemble_type == 'stacking':
            # Use a meta-learner for stacking
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=0.1),
                cv=5,
                n_jobs=-1
            )
        else:
            # Use voting (averaging) for predictions
            ensemble = VotingRegressor(
                estimators=base_models,
                n_jobs=-1
            )
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('ensemble', ensemble)
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info(f"{self.ensemble_type.capitalize()} ensemble model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return f"{self.ensemble_type.capitalize()} Ensemble"


class ModelBuilder:
    """Context class for building models with different strategies"""
    
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: ModelBuildingStrategy):
        """Change the model building strategy"""
        logger.info(f"Switching to {strategy.get_model_name()} strategy")
        self._strategy = strategy
        
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> RegressorMixin:
        """Build and train model using the current strategy"""
        logger.info(f"Building model using {self._strategy.get_model_name()}")
        return self._strategy.build_and_train_model(X_train, y_train, X_val, y_val)
    
    def evaluate_model(self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        predictions = model.predict(X)
        
        metrics = {
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        return metrics
    
    def cross_validate(self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        scores = {
            'r2': cross_val_score(model, X, y, cv=cv, scoring='r2'),
            'neg_mse': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'),
            'neg_mae': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        }
        
        return {
            'cv_r2_mean': scores['r2'].mean(),
            'cv_r2_std': scores['r2'].std(),
            'cv_rmse_mean': np.sqrt(-scores['neg_mse'].mean()),
            'cv_rmse_std': np.sqrt(scores['neg_mse'].std()),
            'cv_mae_mean': -scores['neg_mae'].mean(),
            'cv_mae_std': scores['neg_mae'].std()
        }


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, 
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compare multiple models and return performance metrics"""
    
    strategies = [
        LinearRegressionStrategy(),
        ElasticNetStrategy(),
        EnhancedRandomForestStrategy(),
        GradientBoostingStrategy(),
        XGBoostStrategy(),
        LightGBMStrategy(),
        NeuralNetworkStrategy(),
        EnsembleStrategy(ensemble_type='voting'),
        EnsembleStrategy(ensemble_type='stacking')
    ]
    
    results = []
    
    for strategy in strategies:
        logger.info(f"Training {strategy.get_model_name()}...")
        
        builder = ModelBuilder(strategy)
        model = builder.build_model(X_train, y_train)
        
        train_metrics = builder.evaluate_model(model, X_train, y_train)
        test_metrics = builder.evaluate_model(model, X_test, y_test)
        
        results.append({
            'Model': strategy.get_model_name(),
            'Train R²': train_metrics['r2'],
            'Test R²': test_metrics['r2'],
            'Train RMSE': train_metrics['rmse'],
            'Test RMSE': test_metrics['rmse'],
            'Train MAE': train_metrics['mae'],
            'Test MAE': test_metrics['mae']
        })
    
    return pd.DataFrame(results).sort_values('Test R²', ascending=False)


# Keep the simple LinearRegressionStrategy for backward compatibility
class LinearRegressionStrategy(ModelBuildingStrategy):
    """Basic Linear Regression strategy"""
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Pipeline:
        logger.info("Building Linear Regression model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("Linear Regression model training completed")
        return pipeline
    
    def get_model_name(self) -> str:
        return "Linear Regression"


if __name__ == "__main__":
    logger.info("Advanced Model Building Module Loaded Successfully")
