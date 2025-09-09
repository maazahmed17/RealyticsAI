# 🏠 RealyticsAI - AI-Powered Real Estate Analytics Platform

## Overview

RealyticsAI is a comprehensive real estate analytics platform that combines machine learning with MLflow for production-grade model tracking and management. Currently featuring Bengaluru house price predictions with plans for property recommendations, negotiation agents, and LLM-powered chatbot integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Bengaluru housing dataset (CSV format)

### Installation

```bash
# Clone the repository
cd ~/realyticsAI

# Install dependencies
pip install pandas numpy scikit-learn mlflow fastapi uvicorn pydantic rich requests

# Navigate to backend
cd backend
```

### Running the Application

#### Option 1: Quick Demo (Instant Results)
```bash
python quick_demo.py
```
This runs a complete demonstration with immediate output showing:
- Dataset loading and analysis
- Model training with MLflow tracking
- Sample predictions
- Market analysis
- MLflow tracking information

#### Option 2: Full API Server
```bash
# Start the API server
python main_mlflow.py

# In another terminal, launch MLflow UI
mlflow ui --backend-store-uri file:///home/maaz/.config/zenml/local_stores/05c97d8d-483a-4829-8d7a-797c176c6f95/mlruns

# Access:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - MLflow UI: http://localhost:5000
```

#### Option 3: Interactive CLI
```bash
# Start the server first
python main_mlflow.py &

# Run the CLI demo
python cli_demo.py
```

## 📊 Features

### Current Features
- **Price Prediction**: ML-based house price predictions using Linear Regression
- **Hybrid Model**: Combines ML predictions with similar property analysis (60% ML, 40% similar properties)
- **MLflow Integration**: Complete experiment tracking, model registry, and metrics logging
- **REST API**: FastAPI-based endpoints for predictions and model management
- **Market Analysis**: Insights into price distribution by features and locations
- **Interactive CLI**: User-friendly command-line interface for testing

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server status and metrics |
| `/api/v1/predict` | POST | Make price predictions |
| `/api/v1/model/metrics` | GET | View model performance |
| `/api/v1/train` | POST | Retrain the model |
| `/api/v1/comprehensive-test` | GET | Run full test suite |
| `/api/v1/mlflow/experiments` | GET | List MLflow experiments |
| `/api/v1/mlflow/runs` | GET | View experiment runs |
| `/docs` | GET | Interactive API documentation |

## 🏗️ Architecture

```
realyticsAI/
├── backend/
│   ├── main_mlflow.py          # FastAPI server with MLflow integration
│   ├── quick_demo.py            # Standalone demo script
│   ├── cli_demo.py              # Interactive CLI interface
│   ├── services/
│   │   ├── price_prediction/    # Price prediction ML service
│   │   ├── property_recommendation/  # Future: Recommendation engine
│   │   ├── negotiation_agent/   # Future: Negotiation agent
│   │   └── chatbot_orchestrator/  # Future: LLM chatbot
│   ├── core/                    # Shared components
│   └── api/                     # API routes
├── frontend/                    # Future: React TypeScript UI
├── data/                        # Datasets and models
├── tests/                       # Test suites
└── docs/                        # Documentation
```

## 📈 Model Performance

Current model metrics on Bengaluru dataset (13,320 properties):
- **Test R² Score**: 0.1933
- **Training R² Score**: 0.2108
- **Test MSE**: 17,174.86
- **Features Used**: Number of bathrooms, number of balconies

## 🔬 MLflow Integration

The platform includes comprehensive MLflow integration for:
- **Experiment Tracking**: All training runs are logged
- **Model Registry**: Version control for models
- **Metrics Logging**: Automatic tracking of MSE, R² scores
- **Artifact Storage**: Models and data artifacts saved
- **UI Dashboard**: Visual monitoring at http://localhost:5000

## 📊 Dataset Information

**Bengaluru House Prices Dataset**
- Total Properties: 13,320
- Unique Locations: 1,305
- Price Range: ₹8 - ₹3,600 lakhs
- Average Price: ₹112.57 lakhs
- Median Price: ₹72.00 lakhs

**Top Locations by Property Count:**
1. Whitefield: 540 properties (Avg ₹128.01 lakhs)
2. Sarjapur Road: 399 properties (Avg ₹118.94 lakhs)
3. Electronic City: 302 properties (Avg ₹54.97 lakhs)
4. Kanakpura Road: 273 properties (Avg ₹70.53 lakhs)
5. Thanisandra: 234 properties (Avg ₹82.73 lakhs)

## 🚀 Future Enhancements

1. **Property Recommendation Engine**
   - Content-based filtering
   - Collaborative filtering
   - Hybrid recommendations

2. **Negotiation Agent**
   - AI-powered price negotiation
   - Market trend analysis
   - Optimal pricing strategies

3. **LLM Chatbot Integration**
   - Natural language queries
   - Property search assistance
   - Market insights Q&A

4. **Web UI (React + TypeScript)**
   - Interactive dashboards
   - Real-time predictions
   - Visual analytics

## 🛠️ Development

### Adding New Features
New features should be added as separate services in `backend/services/`. Each service should:
- Have its own pipeline and model management
- Expose APIs through `backend/api/routes/`
- Include comprehensive testing
- Integrate with MLflow for tracking

### Testing
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests
pytest
```

## 📝 License

This project is designed for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
