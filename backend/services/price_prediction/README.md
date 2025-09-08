# 🏠 RealyticsAI Price Prediction Service

A sophisticated real estate price prediction system with MLflow integration, comprehensive market analysis, and intelligent insights powered by machine learning.

## ✨ Features

- **🤖 ML-Powered Predictions**: Advanced price prediction using Linear Regression, Random Forest, and Gradient Boosting models
- **📊 Comprehensive Market Analysis**: Deep insights into property market trends and patterns
- **🔬 MLflow Integration**: Complete experiment tracking and model versioning
- **💡 Intelligent Insights**: Data-driven recommendations and market intelligence
- **🎯 Hybrid Predictions**: Combines ML predictions with historical data for better accuracy
- **🚀 Multiple Modes**: Standalone analysis, API server, interactive predictions, and training modes

## 📋 Requirements

```bash
# Core ML Libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
mlflow>=2.0.0
joblib>=1.0.0

# Visualization & UI
rich>=10.0.0

# API (for server mode)
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
```

## 🚀 Quick Start

### 1. Basic Usage - Complete Analysis

Run the complete analysis with market insights:

```bash
python main.py
```

This will:
- Load the Bengaluru housing dataset
- Train a price prediction model
- Perform comprehensive market analysis
- Display sample predictions
- Generate investment recommendations
- Track everything in MLflow

### 2. API Server Mode

Start the FastAPI server for integration:

```bash
python main.py --mode api
```

Access the API at:
- **API Endpoint**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs

### 3. Interactive Prediction Mode

For interactive property price predictions:

```bash
python main.py --mode predict
```

### 4. Training Mode

Train a specific model type:

```bash
# Linear Regression (default)
python main.py --mode train --model linear

# Random Forest
python main.py --mode train --model random_forest

# Gradient Boosting
python main.py --mode train --model gradient_boost
```

## 📊 Output Example

When you run the main analysis, you'll see:

```
================================================================================
🏠 REALYTICSAI PRICE PREDICTION SYSTEM - COMPLETE ANALYSIS
================================================================================

📂 Loading dataset...
✅ Successfully loaded 13,320 properties
📊 Dataset shape: (13320, 9)
🏘️ Unique locations: 1305

🤖 Training linear model with MLflow tracking...
✅ Model Performance:
   • R² Score: 0.1936
   • RMSE: 131.03
   • MAE: 55.94

📊 COMPREHENSIVE MARKET ANALYSIS
   • Average Price: ₹112.57 Lakhs
   • Price Range: ₹8.0 - 3600.0 Lakhs
   • Top Location: Whitefield

🔮 SAMPLE PREDICTIONS
   • Budget (1 Bath, 0 Balcony): ₹32.83 Lakhs
   • Standard (2 Bath, 1 Balcony): ₹69.11 Lakhs
   • Premium (3 Bath, 2 Balconies): ₹128.92 Lakhs

💡 MARKET INSIGHTS & RECOMMENDATIONS
   • Market shows luxury properties pulling up averages
   • Investment opportunities in: Ananth Nagar, Chandapura
   • High price volatility indicates diverse market
```

## 🔧 Configuration

The system configuration is centralized in the `Config` class within `main.py`:

```python
class Config:
    # Data paths
    BENGALURU_DATA_PATH = "/path/to/bengaluru_house_prices.csv"
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "file:///path/to/mlruns"
    MLFLOW_EXPERIMENT = "realyticsai_price_prediction"
    
    # Model configuration
    MODEL_FEATURES = ["bath", "balcony"]
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
```

## 📈 MLflow Integration

### Viewing Experiments

After running the analysis, view your experiments in MLflow:

```bash
mlflow ui --backend-store-uri file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns
```

Then open: http://localhost:5000

### Tracked Metrics

- **Model Parameters**: features, model type, dataset size
- **Performance Metrics**: R², MSE, RMSE, MAE (for both training and testing)
- **Model Artifacts**: Serialized model, feature importance (if applicable)

## 🏗️ Architecture

```
price_prediction/
├── main.py                 # Main entry point with all functionality
├── README.md              # This file
├── steps/                 # Pipeline components (from original system)
│   ├── data_ingestion_step.py
│   ├── feature_engineering_step.py
│   ├── model_building_step.py
│   └── ...
├── data_splitter.py       # Data splitting utilities
├── feature_engineering.py # Feature engineering methods
├── model_building.py      # Model training logic
└── model_evaluator.py     # Model evaluation metrics
```

## 🎯 Key Components

### 1. PricePredictionSystem Class

The core class that handles:
- Data loading and preprocessing
- Model training with MLflow tracking
- Price predictions (ML and hybrid)
- Market analysis
- Insight generation

### 2. Market Analysis Features

- **Overall Statistics**: Dataset summary, price ranges, location counts
- **Price Distribution**: Quantile-based price segmentation
- **Location Analysis**: Top locations by property count and average prices
- **Feature Analysis**: Price correlation with bathrooms, balconies
- **Correlation Analysis**: Feature importance for price prediction

### 3. Prediction Modes

- **ML Prediction**: Pure machine learning based prediction
- **Hybrid Prediction**: 60% ML + 40% historical average
- **Confidence Scoring**: Based on similar properties found

## 📊 Data Requirements

The system expects a CSV file with the following columns:
- `price`: Property price in lakhs (target variable)
- `bath`: Number of bathrooms
- `balcony`: Number of balconies
- `location`: Property location
- `total_sqft`: Total square footage (optional)
- `bhk`: Number of bedrooms (optional)

## 🔄 Future Integration

This price prediction service is designed to integrate seamlessly with:

1. **Property Recommendation System**: Use price predictions for filtering
2. **Negotiation Agent**: Provide price benchmarks for negotiations
3. **Chatbot Interface**: Natural language queries for price predictions

Integration points:
- REST API endpoints (when running in API mode)
- Direct Python imports of `PricePredictionSystem` class
- MLflow model registry for model serving

## 🐛 Troubleshooting

### Common Issues

1. **Data file not found**:
   - Update `BENGALURU_DATA_PATH` in the Config class
   - Ensure the CSV file exists at the specified location

2. **MLflow tracking issues**:
   - Verify `MLFLOW_TRACKING_URI` path exists
   - Create the directory if it doesn't exist

3. **Low model accuracy**:
   - Consider using more features (extend `MODEL_FEATURES`)
   - Try different model types (random_forest, gradient_boost)
   - Ensure data quality and handle outliers

## 📝 License

Part of the RealyticsAI platform - Property Intelligence System

## 🤝 Contributing

To add new features or models:

1. Extend the `PricePredictionSystem` class
2. Add new model types in the `train_model` method
3. Update the market analysis methods for new insights
4. Ensure MLflow tracking for all experiments

---

**Note**: This is the finalized, production-ready version of the price prediction service. The system is designed for easy data updates and seamless integration with future chatbot/LLM features.
