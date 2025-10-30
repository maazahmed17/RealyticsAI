# backend/api/routes/property_recommendation.py

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Correctly import your RecommendationService
from ...services.property_recommendation.recommender import RecommendationService

router = APIRouter()

# Initialize your recommendation service
# This creates a single instance when the server starts
recommendation_service = RecommendationService()

# Define the input data model for the API
class RecommendationRequest(BaseModel):
    bhk: int = Field(..., example=3, description="Number of bedrooms (BHK)")
    bath: int = Field(..., example=2, description="Number of bathrooms")
    price: float = Field(..., example=100, description="User's budget in INR Lakhs")
    top_n: Optional[int] = Field(5, example=5, description="Number of recommendations to return")

@router.post("/recommend", response_model=List[Dict[str, Any]])
async def get_property_recommendations(request: RecommendationRequest):
    """
    Get property recommendations based on user preferences.
    """
    try:
        # Convert Pydantic model to a dictionary for your service
        preferences = request.dict(exclude_none=True)
        
        # The 'price' field in your recommender is the budget
        preferences['price'] = preferences.pop('price', 100)
        
        top_n = preferences.pop('top_n', 5)

        recommendations = recommendation_service.get_recommendations(
            user_preferences=preferences,
            top_n=top_n
        )
        
        if not recommendations:
            return []
            
        return recommendations
    except Exception as e:
        # Log the error in a real application
        # import logging
        # logging.error(f"Recommendation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

# To test this endpoint directly, you can run this file.
# Note: This requires FastAPI and Uvicorn to be installed.
if __name__ == '__main__':
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/recommendations")

    print("Starting test server for recommendations at http://127.0.0.1:8000/api/v1/recommendations/recommend")
    uvicorn.run(app, host="127.0.0.1", port=8000)