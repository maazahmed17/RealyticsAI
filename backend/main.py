"""
RealyticsAI Backend - Main FastAPI Application
Modular real estate AI platform with price prediction, property recommendations, and negotiation agent
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path

# Import API routes
from api.routes import price_prediction, chatbot, health
# from api.routes import property_recommendation, negotiation_agent  # Coming soon

# Import core components
from core.config.settings import get_settings
from core.database.connection import init_database
from core.models.base import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    settings = get_settings()
    
    # Initialize database
    await init_database()
    
    print("🚀 RealyticsAI Backend Started!")
    print(f"📊 Environment: {settings.environment}")
    print(f"🔧 Debug Mode: {settings.debug}")
    
    yield
    
    # Shutdown
    print("🛑 RealyticsAI Backend Shutting Down...")


# Initialize FastAPI app
app = FastAPI(
    title="RealyticsAI",
    description="AI-Powered Real Estate Platform with Price Prediction, Property Recommendations, and Negotiation Agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["Health"]
)

app.include_router(
    price_prediction.router,
    prefix="/api/v1",
    tags=["Price Prediction"]
)

app.include_router(
    chatbot.router,
    prefix="/api/v1",
    tags=["Chatbot"]
)

# Future feature routes (placeholder)
# app.include_router(
#     property_recommendation.router,
#     prefix="/api/v1",
#     tags=["Property Recommendation"]
# )

# app.include_router(
#     negotiation_agent.router,
#     prefix="/api/v1",
#     tags=["Negotiation Agent"]
# )

# Serve static files for frontend (if built)
static_dir = Path("../frontend/build")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "🏠 Welcome to RealyticsAI Platform!",
        "version": "1.0.0",
        "features": [
            "💰 Price Prediction (Active)",
            "🏡 Property Recommendation (Coming Soon)",
            "🤝 Negotiation Agent (Coming Soon)",
            "🤖 Intelligent Chatbot (In Development)"
        ],
        "docs": "/api/docs",
        "status": "active"
    }


@app.get("/api/v1/status")
async def api_status():
    """API status and feature availability"""
    return {
        "api_version": "v1",
        "status": "healthy",
        "features": {
            "price_prediction": {
                "status": "active",
                "endpoint": "/api/v1/predict",
                "description": "ML-powered property price predictions"
            },
            "property_recommendation": {
                "status": "planned",
                "endpoint": "/api/v1/recommend",
                "description": "Personalized property recommendations"
            },
            "negotiation_agent": {
                "status": "planned", 
                "endpoint": "/api/v1/negotiate",
                "description": "AI-powered negotiation assistance"
            },
            "chatbot": {
                "status": "development",
                "endpoint": "/api/v1/chat",
                "description": "Natural language interface for all features"
            }
        }
    }


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
