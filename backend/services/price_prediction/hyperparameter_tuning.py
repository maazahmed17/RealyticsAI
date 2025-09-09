"""
Hyperparameter Tuning Module for Price Prediction Models
=========================================================
This module implements various hyperparameter optimization techniques:
- Grid Search
- Random Search
- Bayesian Optimization
- Optuna-based optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    KFold, StratifiedKFold
)
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HyperparameterTuner:
    """Advanced hyperparameter tuning for regression models"""
    
    def __init__(self, scoring: str = 'r2', cv: int = 5, n_jobs: int = -1):
        """
        Initialize the hyperparameter tuner
        
        Parameters:
        -----------
        scoring : str
            Scoring metric to optimize ('r2', 'neg_mse', 'neg_mae')
        cv : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs (-1 uses all cores)
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params = {}
        self.best_score = None
        self.search_results = None
        
    def get_xgboost_param_grid(self, search_type: str = 'grid') -> Dict[str, Any]:
        """Get XGBoost parameter grid for tuning"""
        
        if search_type == 'grid':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 1.5]
            }
        else:  # random search
            return {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 0.5),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0.5, 2),
                'min_child_weight': randint(1, 10)
            }
    
    def get_lightgbm_param_grid(self, search_type: str = 'grid') -> Dict[str, Any]:
        """Get LightGBM parameter grid for tuning"""
        
        if search_type == 'grid':
            return {
                'n_estimators': [100, 200, 300],
                'num_leaves': [20, 31, 50],
                'max_depth': [-1, 5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'bagging_freq': [3, 5, 7],
                'lambda_l1': [0, 0.1, 0.5],
                'lambda_l2': [0, 0.1, 0.5]
            }
        else:  # random search
            return {
                'n_estimators': randint(100, 500),
                'num_leaves': randint(20, 100),
                'max_depth': randint(-1, 20),
                'learning_rate': uniform(0.01, 0.3),
                'feature_fraction': uniform(0.5, 0.5),
                'bagging_fraction': uniform(0.5, 0.5),
                'bagging_freq': randint(1, 10),
                'lambda_l1': uniform(0, 1),
                'lambda_l2': uniform(0, 1),
                'min_child_samples': randint(10, 50)
            }
    
    def get_random_forest_param_grid(self, search_type: str = 'grid') -> Dict[str, Any]:
        """Get Random Forest parameter grid for tuning"""
        
        if search_type == 'grid':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        else:  # random search
            return {
                'n_estimators': randint(100, 500),
                'max_depth': [10, 20, 30, 40, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'max_samples': uniform(0.5, 0.5)
            }
    
    def get_gradient_boosting_param_grid(self, search_type: str = 'grid') -> Dict[str, Any]:
        """Get Gradient Boosting parameter grid for tuning"""
        
        if search_type == 'grid':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['sqrt', 'log2', None]
            }
        else:  # random search
            return {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'alpha': uniform(0.8, 0.2)
            }
    
    def grid_search(self, model, param_grid: Dict[str, Any], 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   verbose: int = 1) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform grid search for hyperparameter tuning
        
        Parameters:
        -----------
        model : estimator
            The model to tune
        param_grid : dict
            Parameter grid to search
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        verbose : int
            Verbosity level
            
        Returns:
        --------
        best_model : estimator
            Model with best parameters
        best_params : dict
            Best parameters found
        """
        logger.info("Starting Grid Search...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=verbose,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.search_results = pd.DataFrame(grid_search.cv_results_)
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return grid_search.best_estimator_, self.best_params
    
    def random_search(self, model, param_distributions: Dict[str, Any],
                     X_train: pd.DataFrame, y_train: pd.Series,
                     n_iter: int = 100, verbose: int = 1) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform random search for hyperparameter tuning
        
        Parameters:
        -----------
        model : estimator
            The model to tune
        param_distributions : dict
            Parameter distributions to sample from
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        n_iter : int
            Number of iterations
        verbose : int
            Verbosity level
            
        Returns:
        --------
        best_model : estimator
            Model with best parameters
        best_params : dict
            Best parameters found
        """
        logger.info(f"Starting Random Search with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=verbose,
            refit=True,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.search_results = pd.DataFrame(random_search.cv_results_)
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return random_search.best_estimator_, self.best_params
    
    def bayesian_optimization(self, model_class: str,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            n_trials: int = 100) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform Bayesian optimization using Optuna
        
        Parameters:
        -----------
        model_class : str
            Model class name ('xgboost', 'lightgbm', 'random_forest')
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        best_model : estimator
            Model with best parameters
        best_params : dict
            Best parameters found
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not installed. Using random search instead.")
            return self.auto_tune(model_class, X_train, y_train, method='random')
        
        logger.info(f"Starting Bayesian Optimization with {n_trials} trials...")
        
        def objective(trial):
            if model_class == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_class == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', -1, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_class == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 10, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
                
            else:
                raise ValueError(f"Unknown model class: {model_class}")
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize' if self.scoring in ['r2', 'accuracy'] else 'minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Train final model with best parameters
        if model_class == 'xgboost':
            best_model = xgb.XGBRegressor(**self.best_params, random_state=42, verbosity=0)
        elif model_class == 'lightgbm':
            best_model = lgb.LGBMRegressor(**self.best_params, random_state=42, verbosity=-1)
        elif model_class == 'random_forest':
            best_model = RandomForestRegressor(**self.best_params, random_state=42, n_jobs=-1)
        
        best_model.fit(X_train, y_train)
        
        return best_model, self.best_params
    
    def auto_tune(self, model_class: str,
                 X_train: pd.DataFrame, y_train: pd.Series,
                 method: str = 'random',
                 time_budget: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Automatically tune hyperparameters for a given model
        
        Parameters:
        -----------
        model_class : str
            Model class name ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting')
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        method : str
            Tuning method ('grid', 'random', 'bayesian')
        time_budget : int, optional
            Time budget in seconds
            
        Returns:
        --------
        best_model : estimator
            Model with best parameters
        best_params : dict
            Best parameters found
        """
        logger.info(f"Auto-tuning {model_class} using {method} search...")
        
        # Select model and parameter grid
        if model_class == 'xgboost':
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
            param_grid = self.get_xgboost_param_grid(method)
        elif model_class == 'lightgbm':
            base_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
            param_grid = self.get_lightgbm_param_grid(method)
        elif model_class == 'random_forest':
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = self.get_random_forest_param_grid(method)
        elif model_class == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = self.get_gradient_boosting_param_grid(method)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Perform tuning
        if method == 'grid':
            return self.grid_search(base_model, param_grid, X_train, y_train)
        elif method == 'random':
            n_iter = 50 if time_budget is None else max(10, time_budget // 10)
            return self.random_search(base_model, param_grid, X_train, y_train, n_iter=n_iter)
        elif method == 'bayesian':
            n_trials = 100 if time_budget is None else max(20, time_budget // 5)
            return self.bayesian_optimization(model_class, X_train, y_train, n_trials=n_trials)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def get_tuning_report(self) -> pd.DataFrame:
        """Get detailed tuning report"""
        
        if self.search_results is None:
            return pd.DataFrame()
        
        # Select relevant columns
        report_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        param_cols = [col for col in self.search_results.columns if col.startswith('param_')]
        
        report = self.search_results[report_cols + param_cols].copy()
        report = report.sort_values('rank_test_score')
        
        return report.head(10)
    
    def compare_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                      models: List[str] = None,
                      method: str = 'random') -> pd.DataFrame:
        """
        Compare multiple models with hyperparameter tuning
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        models : list, optional
            List of model names to compare
        method : str
            Tuning method
            
        Returns:
        --------
        comparison_df : pd.DataFrame
            Comparison results
        """
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
        
        results = []
        
        for model_name in models:
            logger.info(f"Tuning {model_name}...")
            
            try:
                best_model, best_params = self.auto_tune(
                    model_name, X_train, y_train, method=method
                )
                
                # Calculate scores
                cv_scores = cross_val_score(
                    best_model, X_train, y_train,
                    cv=self.cv, scoring=self.scoring, n_jobs=-1
                )
                
                results.append({
                    'Model': model_name,
                    'Best Score': self.best_score,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Best Params': str(best_params)
                })
                
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {e}")
                continue
        
        return pd.DataFrame(results).sort_values('Best Score', ascending=False)


def quick_tune(X_train: pd.DataFrame, y_train: pd.Series,
              model_type: str = 'xgboost') -> Tuple[Any, Dict[str, Any]]:
    """
    Quick hyperparameter tuning function
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_type : str
        Type of model to tune
        
    Returns:
    --------
    best_model : estimator
        Tuned model
    best_params : dict
        Best parameters
    """
    tuner = HyperparameterTuner(scoring='r2', cv=5)
    best_model, best_params = tuner.auto_tune(
        model_type, X_train, y_train, method='random'
    )
    
    return best_model, best_params


if __name__ == "__main__":
    logger.info("Hyperparameter Tuning Module loaded successfully")
