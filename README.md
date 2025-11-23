# 🏠 RealyticsAI - Real Estate Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange.svg)](https://ai.google.dev/)

An advanced AI-powered real estate platform that provides intelligent property price predictions, personalized recommendations, and market insights using Machine Learning, Natural Language Processing, and modern web technologies.

---

## ✨ Features

### 🎯 Price Prediction Service
- **XGBoost ML Model**: Accurate property valuation using advanced gradient boosting
- **50+ Engineered Features**: Location encoding, property characteristics, and interaction features
- **Regularization & Early Stopping**: Prevents overfitting with proper validation
- **Performance**: R² Score ~0.85-0.92, MAE ~2.3-3.0 Lakhs
- **Dataset**: 13,000+ Bangalore property listings

### 🔍 Property Recommendation System
- **Content-based Filtering**: Intelligent property matching based on user preferences
- **Multi-criteria Search**: Filter by location, price, BHK, square footage, amenities
- **Smart Location Matching**: Fuzzy search with partial and compound name support
- **Top 5 Best Matches**: Returns the most relevant properties sorted by price

### 🤖 AI Chatbot Interface
- **Gemini AI Integration**: Natural language understanding for property queries
- **Intelligent Intent Detection**: Automatically routes to appropriate service
- **Multi-service Orchestration**: Seamlessly coordinates price prediction and recommendations
- **Conversation Context**: Maintains chat history for better interactions
- **3 Interfaces**: CLI, Web UI, and REST API

### 🌐 Web Interface
- **Modern Responsive Design**: Clean HTML/CSS/JS frontend
- **Real-time Chat**: Interactive chatbot with message history
- **Visual Property Cards**: Displays recommendations with all details
- **Flask + FastAPI**: Dual-server architecture for optimal performance

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CLI Chat   │  │   Web UI     │  │  REST API    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
          ┌──────────────────────────────────────┐
          │    Gemini AI Intelligent Router      │
          │  (Intent Detection & Classification)  │
          └──────────┬───────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌───────────┐ ┌────────────┐ ┌──────────────┐
│   Price   │ │ Property   │ │   Market     │
│ Prediction│ │Recommender │ │  Analysis    │
└─────┬─────┘ └─────┬──────┘ └──────┬───────┘
      │             │               │
      ▼             ▼               ▼
┌─────────────────────────────────────────┐
│         Data Layer & ML Models          │
│  • XGBoost Model                        │
│  • Bengaluru Housing Dataset            │
│  • Feature Engineering Pipeline         │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/maazahmed17/RealyticsAI.git
   cd RealyticsAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

4. **Run the system**

   **Option A: Interactive CLI Chat**
   ```bash
   python run_unified_system.py
   # Select option 1
   ```

   **Option B: Web Interface**
   ```bash
   python run_unified_system.py
   # Select option 2
   # Access at: http://localhost:5000
   ```

   **Option C: Direct Python Script**
   ```bash
   python realytics_ai.py
   ```

---

## 💻 Usage Examples

### Price Prediction
```
You: What's the price of a 3 BHK apartment in Whitefield with 2 bathrooms?

Bot: Based on the property details you provided:
     🏠 Location: Whitefield
     🛏️ BHK: 3
     🛁 Bathrooms: 2
     
     💰 Estimated Price: ₹125.50 Lakhs
     
     This is based on analysis of similar properties in the area.
```

### Property Recommendations
```
You: Show me apartments under 50 lakhs in Marathahalli

Bot: Here are 5 properties matching your criteria:
     
     1. 📍 Marathahalli - 2 BHK, 1200 sqft - ₹42.0 Lakhs
     2. 📍 Marathahalli - 1 BHK, 800 sqft - ₹38.5 Lakhs
     3. 📍 Marathahalli - 2 BHK, 1000 sqft - ₹45.0 Lakhs
     ...
```

### Market Analysis
```
You: What's the average price in HSR Layout?

Bot: 📊 HSR Layout Market Overview:
     • Average Price: ₹85.30 Lakhs
     • Price Range: ₹25.0L - ₹350.0L
     • Total Properties: 542
     • Popular BHK: 2-3 BHK
```

---

## 📁 Project Structure

```
RealyticsAI/
├── backend/                    # Backend services
│   ├── core/                   # Core configuration
│   │   ├── config/            # Settings and environment
│   │   ├── database/          # Database connections
│   │   └── models/            # Data models
│   └── services/              # Microservices
│       ├── chatbot_orchestrator/    # AI chatbot service
│       ├── price_prediction/        # ML price prediction
│       ├── property_recommendation/ # Recommendation engine
│       └── recommendation_service/  # Additional rec features
├── frontend/                  # Web interface
│   ├── index.html            # Main web page
│   ├── styles.css            # UI styling
│   ├── App.js                # Frontend logic
│   └── src/                  # React components (optional)
├── src/                       # ML training code
│   ├── models/               # Model implementations
│   │   ├── feature_engineering_advanced.py
│   │   └── xgboost_strategy.py
│   ├── train_enhanced_model.py     # Main training script
│   ├── evaluate_fixed_model.py     # Model evaluation
│   └── chatbot.py                  # Chatbot implementation
├── data/                      # Data directory
│   ├── raw/                  # Raw datasets
│   │   └── bengaluru_house_prices.csv
│   ├── processed/            # Processed data
│   └── models/               # Trained models
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── negotiation/          # Negotiation tests
│   └── e2e/                  # End-to-end tests
├── config/                    # Configuration files
│   └── settings.py           # Settings module
├── docs/                      # Documentation
│   └── archive/              # Development notes archive
├── realytics_ai.py           # Main chatbot application
├── run_unified_system.py     # System launcher
├── web_server.py             # Flask web server
├── get_xgboost_metrics.py    # Model metrics utility
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.0-flash

# Optional: Data paths
DATA_PATH=/path/to/bengaluru_house_prices.csv
MODEL_DIR=/path/to/models
```

### Model Training

Train or retrain the XGBoost model:

```bash
cd src/
python train_enhanced_model.py --no-tuning
```

Evaluate model performance:

```bash
cd src/
python evaluate_fixed_model.py
```

Get detailed metrics:

```bash
python get_xgboost_metrics.py
```

---

## 🛠️ API Reference

### Flask Web Server (Port 5000)

#### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What's the price of a 2 BHK in Koramangala?"
}
```

**Response:**
```json
{
  "response": "Based on your query...",
  "success": true,
  "service_used": "price_prediction",
  "has_prediction": true,
  "property_details": { ... },
  "recommendations": [ ... ]
}
```

#### Reset Chat
```http
POST /api/reset
```

#### Analyze Intent (Debug)
```http
POST /api/analyze
Content-Type: application/json

{
  "query": "Show me properties in Whitefield"
}
```

### FastAPI Backend (Port 8000)

FastAPI provides additional endpoints for advanced features. Documentation available at `http://localhost:8000/docs` when running.

---

## 📊 Model Performance

### XGBoost Price Prediction Model

| Metric | Train | Test |
|--------|-------|------|
| **R² Score** | 0.88-0.92 | 0.85-0.90 |
| **RMSE (Lakhs)** | 2.0-2.5 | 2.3-3.0 |
| **MAE (Lakhs)** | 1.5-2.0 | 1.8-2.5 |
| **MAPE (%)** | 8-12% | 10-15% |

**Key Improvements:**
- ✅ No data leakage (removed `price_per_sqft` feature)
- ✅ Proper regularization (L1/L2 penalties)
- ✅ Early stopping with validation set
- ✅ Robust location encoding (frequency + smoothed price)
- ✅ 50+ engineered features including interactions

**Dataset:**
- 13,000+ Bangalore properties
- Features: Location, BHK, Total Sqft, Bath, Balcony, Age, Floor, Parking
- Target: Price in Lakhs

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Suite
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Negotiation tests
pytest tests/negotiation/
```

### Manual Testing
```bash
# Test price prediction
python tests/unit/test_xgboost_prediction.py

# Test chatbot
python realytics_ai.py
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Ensure you're in the project root and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Gemini API key not found"
**Solution:** Set your API key in the `.env` file:
```bash
GEMINI_API_KEY=your_actual_api_key
```

### Issue: "Data file not found"
**Solution:** Ensure the dataset is in the correct location:
```bash
data/raw/bengaluru_house_prices.csv
# or
data/bengaluru_house_prices.csv
```

### Issue: Model predictions are inaccurate
**Solution:** Retrain the model with the latest data:
```bash
cd src/
python train_enhanced_model.py
```

### Issue: Web server won't start
**Solution:** Check if ports 5000 and 8000 are available:
```bash
lsof -i :5000
lsof -i :8000
# Kill processes if needed
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Write/update tests**
5. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Write unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

---

## 📚 Technology Stack

### Machine Learning & Data Science
- **XGBoost** - Gradient boosting for price prediction
- **LightGBM** - Alternative gradient boosting implementation
- **Scikit-learn** - Feature engineering and preprocessing
- **Pandas & NumPy** - Data manipulation and analysis

### AI & NLP
- **Google Gemini AI** - Natural language understanding and intent detection
- **Transformers** - Advanced NLP capabilities (optional)
- **TensorFlow & PyTorch** - Deep learning frameworks

### Web Frameworks
- **Flask** - Web server and frontend serving
- **FastAPI** - High-performance REST API
- **Uvicorn** - ASGI server for FastAPI

### Development Tools
- **Rich** - Beautiful terminal output
- **MLflow** - Experiment tracking
- **Pytest** - Testing framework
- **Black** - Code formatting

---

## 📈 Roadmap

### Current Features (v1.0)
- ✅ Price prediction with XGBoost
- ✅ Property recommendations
- ✅ AI chatbot with Gemini integration
- ✅ Web interface
- ✅ CLI interface
- ✅ REST API

### Planned Features (v2.0)
- 🔄 User authentication and profiles
- 🔄 Save favorite properties
- 🔄 Email notifications for price changes
- 🔄 Advanced market trend analysis
- 🔄 Property comparison tool
- 🔄 Image recognition for property features
- 🔄 Mobile app (React Native)

### Future Enhancements (v3.0)
- 🔮 Multi-city support (Delhi, Mumbai, etc.)
- 🔮 Rental price predictions
- 🔮 Investment ROI calculator
- 🔮 Neighborhood safety scores
- 🔮 School district ratings
- 🔮 Voice interface support

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 RealyticsAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👥 Authors & Contact

**Maaz Ahmed**
- GitHub: [@maazahmed17](https://github.com/maazahmed17)
- Email: [Your Email Here]

---

## 🙏 Acknowledgments

- **Bengaluru Housing Dataset** - Data source for training
- **Google Gemini AI** - Natural language processing
- **XGBoost Team** - ML library
- **Scikit-learn Contributors** - Feature engineering tools
- **Flask & FastAPI Communities** - Web framework support

---

## 📝 Citations

If you use this project in your research or work, please cite:

```bibtex
@software{realyticsai2024,
  title = {RealyticsAI: Real Estate Intelligence Platform},
  author = {Ahmed, Maaz},
  year = {2024},
  url = {https://github.com/maazahmed17/RealyticsAI}
}
```

---

## 🔗 Links

- **Repository:** https://github.com/maazahmed17/RealyticsAI
- **Issues:** https://github.com/maazahmed17/RealyticsAI/issues
- **Wiki:** https://github.com/maazahmed17/RealyticsAI/wiki
- **Gemini AI:** https://ai.google.dev/

---

<div align="center">

**Made with ❤️ by the RealyticsAI Team**

⭐ Star us on GitHub if you find this project useful!

[Report Bug](https://github.com/maazahmed17/RealyticsAI/issues) · [Request Feature](https://github.com/maazahmed17/RealyticsAI/issues) · [Documentation](https://github.com/maazahmed17/RealyticsAI/wiki)

</div>
