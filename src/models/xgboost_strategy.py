"""XGBoost strategy for price prediction"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

class XGBoostStrategy:
    """XGBoost Regressor with optimized hyperparameters"""
    
    def __init__(self, **kwargs):
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 1000),
            'max_depth': kwargs.get('max_depth', 5),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.7),
            'gamma': kwargs.get('gamma', 0.1),
            'reg_alpha': kwargs.get('reg_alpha', 0.5),
            'reg_lambda': kwargs.get('reg_lambda', 1.0),
            'min_child_weight': kwargs.get('min_child_weight', 3),
            'random_state': kwargs.get('random_state', 42),
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'eval_metric': 'rmse'
        }
    
    def build_and_train_model(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None) -> Pipeline:
        """Build and train the XGBoost model"""
        logger.info("Building XGBoost model...")
        
        # Create base model
        model = xgb.XGBRegressor(**self.params)
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def get_model_name(self) -> str:
        return "XGBoost"