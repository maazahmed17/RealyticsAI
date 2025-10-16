"""
RealyticsAI Backend - WITH MLFLOW INTEGRATION
Complete ML Operations tracking with MLflow and ZenML
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import os
import sys
import uvicorn
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import pickle

# Add the price prediction system to path
sys.path.append('/home/maaz/prices-predictor-system')

# Initialize FastAPI app
app = FastAPI(
    title="RealyticsAI - MLflow Enhanced",
    description="Price Prediction with MLflow Experiment Tracking",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow Configuration
MLFLOW_TRACKING_URI = "file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("realyticsai_bengaluru_experiment")

# Load Bengaluru dataset
BENGALURU_DATA_PATH = "/home/maaz/RealyticsAI/data/bengaluru_house_prices.csv"
bengaluru_data = None
current_model = None
model_metrics = {}

# Load data on startup
def load_bengaluru_data():
    global bengaluru_data
    try:
        if os.path.exists(BENGALURU_DATA_PATH):
            bengaluru_data = pd.read_csv(BENGALURU_DATA_PATH)
            print(f"✅ Loaded Bengaluru dataset: {bengaluru_data.shape[0]} properties")
            print(f"📊 Columns: {list(bengaluru_data.columns)}")
            print(f"💰 Price range: ₹{bengaluru_data['price'].min():.1f} - ₹{bengaluru_data['price'].max():.1f} lakhs")
            return True
        else:
            print(f"❌ Data file not found: {BENGALURU_DATA_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

# Train model with MLflow tracking
def train_model_with_mlflow():
    global current_model, model_metrics, bengaluru_data
    
    if bengaluru_data is None:
        return False
    
    try:
        with mlflow.start_run(run_name="bengaluru_price_model"):
            # Log data info
            mlflow.log_param("dataset", "bengaluru_house_prices")
            mlflow.log_param("total_properties", len(bengaluru_data))
            mlflow.log_param("features", ["bath", "balcony"])
            mlflow.log_param("model_type", "LinearRegression")
            
            # Prepare data
            features = bengaluru_data[['bath', 'balcony']].fillna(bengaluru_data[['bath', 'balcony']].mean())
            target = bengaluru_data['price']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_predictions)
            test_mse = mean_squared_error(y_test, test_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            
            # Log metrics
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="bengaluru_price_predictor"
            )
            
            # Store model and metrics
            current_model = model
            model_metrics = {
                "train_mse": round(train_mse, 4),
                "test_mse": round(test_mse, 4),
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Model trained successfully!")
            print(f"📊 R² Score: {test_r2:.4f}")
            print(f"📈 MSE: {test_mse:.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

# Pydantic models
class PropertyFeatures(BaseModel):
    bath: Optional[int] = 2
    balcony: Optional[int] = 1
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    location: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR Lakhs"
    model_type: str
    confidence_score: Optional[float]
    similar_properties_count: int
    price_range: Dict[str, float]
    market_insights: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Load data and train model on startup"""
    print("🚀 Starting RealyticsAI with MLflow Integration...")
    
    # Load data
    if load_bengaluru_data():
        # Train model with MLflow tracking
        train_model_with_mlflow()
    
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print("🎯 RealyticsAI MLflow Backend Ready!")

@app.get("/")
async def root():
    """Root endpoint with MLflow info"""
    return {
        "message": "🏠 RealyticsAI with MLflow Integration",
        "status": "active",
        "features": {
            "mlflow_tracking": True,
            "experiment_tracking": True,
            "model_registry": True,
            "zenml_pipeline": "compatible"
        },
        "dataset": "Bengaluru Housing Prices" if bengaluru_data is not None else "Not loaded",
        "properties_count": len(bengaluru_data) if bengaluru_data is not None else 0,
        "model_trained": current_model is not None,
        "model_metrics": model_metrics,
        "mlflow_ui": "Run: mlflow ui --backend-store-uri file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns",
        "api_docs": "/api/docs"
    }

@app.get("/api/v1/mlflow/experiments")
async def get_experiments():
    """Get MLflow experiments"""
    try:
        experiments = mlflow.list_experiments()
        return {
            "experiments": [
                {
                    "name": exp.name,
                    "experiment_id": exp.experiment_id,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": exp.tags
                }
                for exp in experiments
            ],
            "tracking_uri": MLFLOW_TRACKING_URI
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/mlflow/runs")
async def get_runs():
    """Get MLflow runs for current experiment"""
    try:
        experiment = mlflow.get_experiment_by_name("realyticsai_bengaluru_experiment")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return {
                "experiment": experiment.name,
                "total_runs": len(runs),
                "runs": runs.head(10).to_dict('records') if not runs.empty else []
            }
        return {"message": "No experiment found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_price(features: PropertyFeatures):
    """
    Predict property price with MLflow model tracking
    """
    if bengaluru_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        # Log prediction request in MLflow
        with mlflow.start_run(run_name="prediction_request", nested=True):
            bath = features.bath or features.bathrooms or 2
            balcony = features.balcony or 1
            
            mlflow.log_param("bath", bath)
            mlflow.log_param("balcony", balcony)
            if features.location:
                mlflow.log_param("location", features.location)
            
            # Use trained model if available
            if current_model is not None:
                # Model prediction
                input_features = pd.DataFrame([[bath, balcony]], columns=['bath', 'balcony'])
                ml_prediction = current_model.predict(input_features)[0]
                
                # Also get similar properties for comparison
                similar_properties = bengaluru_data[
                    (bengaluru_data['bath'] == bath) & 
                    (bengaluru_data['balcony'] == balcony)
                ]
                
                if len(similar_properties) > 0:
                    avg_price = similar_properties['price'].mean()
                    min_price = similar_properties['price'].min()
                    max_price = similar_properties['price'].max()
                    
                    # Weighted prediction (60% ML model, 40% similar properties average)
                    final_prediction = 0.6 * ml_prediction + 0.4 * avg_price
                    
                    mlflow.log_metric("ml_prediction", ml_prediction)
                    mlflow.log_metric("similar_avg", avg_price)
                    mlflow.log_metric("final_prediction", final_prediction)
                    
                    # Calculate confidence based on R² and number of similar properties
                    confidence = min(0.95, model_metrics.get("test_r2", 0.5) + 
                                   (min(len(similar_properties), 100) / 1000))
                    
                    return PredictionResponse(
                        predicted_price=round(final_prediction, 2),
                        currency="INR Lakhs",
                        model_type="ML + Similar Properties Hybrid",
                        confidence_score=round(confidence, 2),
                        similar_properties_count=len(similar_properties),
                        price_range={
                            "min": round(min_price, 2),
                            "max": round(max_price, 2),
                            "average": round(avg_price, 2),
                            "ml_prediction": round(ml_prediction, 2)
                        },
                        market_insights={
                            "bath": bath,
                            "balcony": balcony,
                            "model_r2_score": model_metrics.get("test_r2", 0),
                            "market_position": "below_average" if final_prediction < avg_price else "above_average"
                        }
                    )
            
            # Fallback to similar properties only
            similar_properties = bengaluru_data[
                (bengaluru_data['bath'] == bath) & 
                (bengaluru_data['balcony'] == balcony)
            ]
            
            if len(similar_properties) > 0:
                avg_price = similar_properties['price'].mean()
                min_price = similar_properties['price'].min()
                max_price = similar_properties['price'].max()
                
                mlflow.log_metric("predicted_price", avg_price)
                
                return PredictionResponse(
                    predicted_price=round(avg_price, 2),
                    currency="INR Lakhs",
                    model_type="Similar Properties Average",
                    confidence_score=None,
                    similar_properties_count=len(similar_properties),
                    price_range={
                        "min": round(min_price, 2),
                        "max": round(max_price, 2),
                        "average": round(avg_price, 2)
                    },
                    market_insights={
                        "bath": bath,
                        "balcony": balcony
                    }
                )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/train")
async def retrain_model():
    """Retrain model with MLflow tracking"""
    if bengaluru_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    success = train_model_with_mlflow()
    
    if success:
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": model_metrics,
            "mlflow_tracking": True
        }
    else:
        raise HTTPException(status_code=500, detail="Model training failed")

@app.get("/api/v1/model/metrics")
async def get_model_metrics():
    """Get current model metrics"""
    return {
        "model_trained": current_model is not None,
        "metrics": model_metrics,
        "mlflow_experiment": "realyticsai_bengaluru_experiment",
        "tracking_uri": MLFLOW_TRACKING_URI
    }

@app.get("/api/v1/comprehensive-test")
async def comprehensive_test():
    """Run comprehensive test with MLflow tracking"""
    if bengaluru_data is None or current_model is None:
        raise HTTPException(status_code=500, detail="Model not ready")
    
    test_cases = [
        {"bath": 1, "balcony": 0, "description": "1 Bath, No Balcony (Budget)"},
        {"bath": 2, "balcony": 1, "description": "2 Bath, 1 Balcony (Standard)"},
        {"bath": 3, "balcony": 2, "description": "3 Bath, 2 Balconies (Premium)"},
        {"bath": 4, "balcony": 3, "description": "4 Bath, 3 Balconies (Luxury)"}
    ]
    
    with mlflow.start_run(run_name="comprehensive_test"):
        results = []
        
        for test in test_cases:
            features = PropertyFeatures(bath=test["bath"], balcony=test["balcony"])
            prediction = await predict_price(features)
            
            mlflow.log_metric(f"price_{test['bath']}bath_{test['balcony']}balcony", 
                            prediction.predicted_price)
            
            results.append({
                "description": test["description"],
                "bath": test["bath"],
                "balcony": test["balcony"],
                "predicted_price": prediction.predicted_price,
                "ml_prediction": prediction.price_range.get("ml_prediction"),
                "confidence": prediction.confidence_score,
                "similar_properties": prediction.similar_properties_count,
                "price_range": prediction.price_range
            })
        
        # Market analysis
        price_by_bath = bengaluru_data.groupby('bath')['price'].agg(['mean', 'median', 'count'])
        price_by_balcony = bengaluru_data.groupby('balcony')['price'].agg(['mean', 'median', 'count'])
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "model_performance": model_metrics,
            "test_predictions": results,
            "market_analysis": {
                "total_properties": len(bengaluru_data),
                "overall_average": round(bengaluru_data['price'].mean(), 2),
                "overall_median": round(bengaluru_data['price'].median(), 2),
                "price_by_bathrooms": price_by_bath.to_dict(),
                "price_by_balconies": price_by_balcony.to_dict()
            },
            "mlflow_tracking": {
                "experiment": "realyticsai_bengaluru_experiment",
                "tracking_enabled": True,
                "ui_command": "mlflow ui --backend-store-uri file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns"
            }
        }

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 REALYTICSAI WITH MLFLOW INTEGRATION")
    print("=" * 60)
    print("📊 Starting server with ML Operations tracking...")
    print("🔧 MLflow UI: Run the following command in another terminal:")
    print("   mlflow ui --backend-store-uri file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns")
    print("=" * 60)
    
    uvicorn.run(
        "main_mlflow:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
