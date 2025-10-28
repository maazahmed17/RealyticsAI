# RealyticsAI - AI Agent Instructions

## Project Overview
RealyticsAI is a real estate intelligence platform combining ML-powered price prediction, property recommendations, and market analysis through a unified interface. The system uses MLflow for experiment tracking and model deployment.

## Critical Knowledge

### Architecture & Component Boundaries
- Core ML code lives ONLY in `src/models/` (single source of truth)
- Raw data lives ONLY in `data/raw/` (single source of truth)
- Backend microservices in `backend/services/`:
  - Price Prediction: ML-based property valuation
  - Property Recommendation: Intelligent property search
  - Chatbot Orchestrator: Smart query routing using Gemini AI

### Data Flow & Integration
1. Raw data ingested from `data/raw/bengaluru_house_prices.csv`
2. Features engineered without data leakage (NO price-derived features)
3. Models trained with MLflow tracking
4. Predictions served through FastAPI endpoints
5. Gemini AI routes user queries to appropriate services

### Key Development Workflows

#### Training New Models
```bash
cd /home/maaz/RealyticsAI/src
python train_enhanced_model.py --no-tuning  # Uses clean feature engineering
python evaluate_fixed_model.py  # Validates model performance
```

#### Model Deployment Files
- Model: `data/models/xgboost_fixed_*.pkl`
- Scaler: `data/models/scaler_*.pkl`
- Features: `data/models/feature_columns_*.pkl`
- Metrics: `data/models/fixed_model_metrics.json`

### Project-Specific Conventions

#### Feature Engineering Rules
- NEVER create features using the `price` column
- NO location encoding using price statistics
- ALWAYS use validation sets for early stopping
- Monitor train/test gap more than absolute scores

#### Model Performance Expectations
- R² should be ~0.77 (not 0.99 which indicates leakage)
- Train/Test gap ratio should be ~1.01x
- RMSE around 12L is expected and correct

### Common Issues & Solutions

1. If predictions stuck at ₹1.85 Crores:
   - Verify model files in `data/models/xgboost_fixed_*.pkl`
   - Check data file accessibility
   - Restart chatbot service

2. Feature mismatch warnings:
   - Expected on first run (initialization)
   - Should resolve after first prediction
   - Verify all 25 features are generated

### Testing & Validation
- Test predictions with `evaluate_fixed_model.py`
- Verify NO data leakage (price should not be perfect)
- Sample test query: "What's the price of a 3BHK in Whitefield?"
- Run `python run_unified_system.py` for integrated testing

### Required Dependencies
```bash
pip install xgboost lightgbm scikit-learn pandas numpy rich mlflow
```