# RealyticsAI - Real Estate Intelligence Platform

## Project Structure

```
RealyticsAI/
├── backend/              # Backend services
│   ├── core/            # Core configuration
│   └── services/        # Microservices (chatbot, recommendation)
├── data/                # Data directory
│   ├── raw/            # Raw data files (SINGLE SOURCE OF TRUTH)
│   ├── processed/      # Processed data
│   └── models/         # Trained models
├── frontend/           # Web UI (future)
├── src/                # ML Training Code (SINGLE SOURCE OF TRUTH)
│   ├── models/         # ML algorithms and feature engineering
│   └── *.py           # Training scripts
└── tests/             # Test files
```

## Quick Start - Model Training

### 1. Train the Fixed Model (No Data Leakage)

```bash
cd /home/maaz/RealyticsAI/src
python train_enhanced_model.py --no-tuning
```

This will:
- Load data from `data/raw/bengaluru_house_prices.csv`
- Apply CLEAN feature engineering (no data leakage)
- Train with proper regularization and early stopping
- Use validation set to prevent overfitting
- Save the best model to `data/models/`

### 2. Evaluate the New Fixed Model

```bash
cd /home/maaz/RealyticsAI/src
python evaluate_fixed_model.py  # Evaluates NEW fixed model
```

**Note:** The old `calculate_metrics.py` evaluates the OLD overfit model. Use `evaluate_fixed_model.py` for the new clean model.

### 3. Expected Results

**Healthy Model Indicators:**
- Train R² and Test R² within 0.05 of each other
- Both R² scores between 0.85 - 0.92
- RMSE Train/Test ratio < 1.3x
- No R² above 0.95 (would indicate overfitting)

## What Was Fixed?

See [OVERFITTING_FIXES_APPLIED.md](./OVERFITTING_FIXES_APPLIED.md) for complete details.

**Summary:**
1. ✅ Removed data leakage (`price_per_sqft` and price-based location encoding)
2. ✅ Added proper regularization (L1/L2, reduced tree depth)
3. ✅ Implemented early stopping with validation set
4. ✅ Cleaned up duplicate code and data files

## Three Features

This system combines three AI features:

1. **Price Prediction** (Fixed & Production-Ready)
   - XGBoost with 50+ engineered features
   - No data leakage
   - Proper validation

2. **Property Recommendation** (Working)
   - Content-based filtering
   - Location: `backend/services/recommendation_service/`

3. **Negotiation Agent** (Planned)
   - AI-powered negotiation assistance

## Data

**Single Source:** `data/raw/bengaluru_house_prices.csv`
- Contains ~13,000 Bangalore property listings
- Features: location, size, sqft, bathrooms, balconies, price

## Model Performance

| Metric | Before Fix | After Fix (Expected) |
|--------|------------|---------------------|
| Train R² | 0.9998 ❌ | 0.88-0.92 ✅ |
| Test R² | ~0.85 | 0.85-0.90 ✅ |
| Train RMSE | 0.55L | 2.0-2.5L |
| Test RMSE | 2.89L | 2.3-3.0L |
| Gap Ratio | 5.25x ❌ | 1.1-1.3x ✅ |

## Important Paths

- **Data:** `/home/maaz/RealyticsAI/data/raw/bengaluru_house_prices.csv`
- **Models:** `/home/maaz/RealyticsAI/data/models/`
- **ML Code:** `/home/maaz/RealyticsAI/src/models/`
- **Training:** `/home/maaz/RealyticsAI/src/train_enhanced_model.py`

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- xgboost
- lightgbm
- scikit-learn
- pandas
- numpy
- rich (for pretty CLI output)
- mlflow (for experiment tracking)

## Notes

- **DO NOT** create features using the `price` column
- **ALWAYS** use validation sets for early stopping
- **MONITOR** train/test gap more than absolute scores
- **KEEP** single source of truth for code and data
