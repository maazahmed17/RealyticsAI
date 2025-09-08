"""
Price Prediction API Routes
Handles property price prediction requests and model management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from services.price_prediction.price_predictor import PricePredictionService
from core.config.settings import get_settings

router = APIRouter()

# Initialize price prediction service
price_service = PricePredictionService()


# Pydantic models for API
class PropertyFeatures(BaseModel):
    """Property features for price prediction"""
    bedrooms: Optional[int] = Field(None, ge=1, le=10, description="Number of bedrooms")
    bathrooms: Optional[int] = Field(None, ge=1, le=10, description="Number of bathrooms") 
    balconies: Optional[int] = Field(None, ge=0, le=5, description="Number of balconies")
    area: Optional[float] = Field(None, gt=0, description="Property area in sq ft")
    location: Optional[str] = Field(None, description="Property location")
    year_built: Optional[int] = Field(None, ge=1900, le=2025, description="Year built")
    property_type: Optional[str] = Field(None, description="Type of property")
    
    # Bengaluru specific features (based on your data)
    bath: Optional[int] = Field(None, ge=1, le=10, description="Number of bathrooms (alternative field)")
    balcony: Optional[int] = Field(None, ge=0, le=5, description="Number of balconies (alternative field)")


class PredictionRequest(BaseModel):
    """Price prediction request"""
    property_features: PropertyFeatures
    model_type: Optional[str] = Field("local", description="Model type: 'local' or 'ames'")
    return_confidence: bool = Field(False, description="Include prediction confidence interval")


class PredictionResponse(BaseModel):
    """Price prediction response"""
    predicted_price: float = Field(description="Predicted property price")
    currency: str = Field("INR Lakhs", description="Currency/unit of the price")
    model_used: str = Field(description="ML model used for prediction")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval if requested")
    similar_properties: Optional[List[Dict]] = Field(None, description="Similar properties for reference")
    market_insights: Optional[Dict[str, Any]] = Field(None, description="Market analysis insights")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    model_name: str
    status: str
    last_trained: Optional[str]
    performance_metrics: Optional[Dict[str, float]]
    features_expected: List[str]


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict property price based on features
    
    Returns predicted price using trained ML models
    """
    try:
        # Prepare features for prediction
        features = request.property_features.dict(exclude_none=True)
        
        # Map common field names
        if 'bedrooms' in features and 'bath' not in features:
            features['bath'] = features.get('bathrooms', features.get('bedrooms', 2))
        if 'balconies' in features and 'balcony' not in features:
            features['balcony'] = features.get('balconies', 1)
        
        # Get prediction from service
        result = await price_service.predict_price(
            features=features,
            model_type=request.model_type,
            include_confidence=request.return_confidence
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch")
async def predict_batch_prices(properties: List[PropertyFeatures], model_type: str = "local"):
    """
    Predict prices for multiple properties at once
    """
    try:
        predictions = []
        
        for prop in properties:
            features = prop.dict(exclude_none=True)
            
            # Map field names
            if 'bedrooms' in features and 'bath' not in features:
                features['bath'] = features.get('bathrooms', features.get('bedrooms', 2))
            if 'balconies' in features and 'balcony' not in features:
                features['balcony'] = features.get('balconies', 1)
            
            result = await price_service.predict_price(
                features=features,
                model_type=model_type,
                include_confidence=False
            )
            
            predictions.append({
                "input_features": features,
                "predicted_price": result["predicted_price"],
                "currency": result["currency"]
            })
        
        return {
            "batch_size": len(predictions),
            "predictions": predictions,
            "model_used": model_type
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/train")
async def train_model(
    data_file: UploadFile = File(...),
    target_column: str = Form(...),
    model_name: str = Form("custom_model")
):
    """
    Train a new model with uploaded data
    """
    try:
        # Validate file type
        settings = get_settings()
        file_ext = Path(data_file.filename).suffix.lower()
        
        if file_ext not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed types: {settings.allowed_file_types}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await data_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Train model with uploaded data
            result = await price_service.train_model(
                data_file_path=tmp_file_path,
                target_column=target_column,
                model_name=model_name
            )
            
            return {
                "message": "Model training completed successfully",
                "model_name": model_name,
                "training_results": result
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )


@router.get("/models", response_model=List[ModelStatusResponse])
async def list_models():
    """
    List all available trained models
    """
    try:
        models = await price_service.list_available_models()
        return models
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_name}/status", response_model=ModelStatusResponse)
async def get_model_status(model_name: str):
    """
    Get detailed status of a specific model
    """
    try:
        status = await price_service.get_model_status(model_name)
        return status
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found or status unavailable"
        )


@router.get("/market-analysis/{location}")
async def get_market_analysis(location: str, property_type: Optional[str] = None):
    """
    Get market analysis for a specific location
    """
    try:
        analysis = await price_service.get_market_analysis(location, property_type)
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Market analysis failed: {str(e)}"
        )


@router.get("/similar-properties")
async def find_similar_properties(
    bedrooms: int,
    bathrooms: int,
    location: Optional[str] = None,
    max_results: int = 5
):
    """
    Find similar properties for comparison
    """
    try:
        similar = await price_service.find_similar_properties(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            location=location,
            max_results=max_results
        )
        
        return {
            "query": {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location
            },
            "similar_properties": similar,
            "count": len(similar)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar properties search failed: {str(e)}"
        )
