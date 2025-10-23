import logging
import sys
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from zenml import ArtifactConfig, step
from zenml.client import Client

# Add parent directory to path to import model_building_advanced
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_building_advanced import XGBoostStrategy, ModelBuilder

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains an XGBoost model for improved prediction accuracy.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained XGBoost pipeline with preprocessing.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging for scikit-learn/xgboost
        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()

        logging.info("Building and training the XGBoost model with advanced parameters...")
        
        # Import XGBoost directly to avoid double scaling
        import xgboost as xgb
        
        # Create XGBoost model with optimized parameters
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            verbosity=0
        )
        
        # Create pipeline with preprocessor and XGBoost
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ])
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        
        logging.info("XGBoost model training completed.")
        logging.info(f"Model type: {type(pipeline.named_steps['model']).__name__}")

        # Log expected columns
        if len(categorical_cols) > 0:
            onehot_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
            onehot_encoder.fit(X_train[categorical_cols])
            expected_columns = numerical_cols.tolist() + list(
                onehot_encoder.get_feature_names_out(categorical_cols)
            )
        else:
            expected_columns = numerical_cols.tolist()
            
        logging.info(f"Model expects {len(expected_columns)} columns")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline
