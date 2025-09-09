# 🤖 RealyticsAI Gemini Integration

## Overview
RealyticsAI now features a powerful natural language interface powered by Google's Gemini 1.5 Flash API. This integration enables users to interact with the real estate system using conversational queries and receive intelligent, context-aware responses.

## ✨ Features

### 1. **Natural Language Price Prediction**
- Ask questions in plain English about property prices
- Automatic extraction of property features from queries
- Detailed explanations of predictions with market context

### 2. **Intelligent Market Analysis**
- Real-time market insights based on Bengaluru data
- Trend analysis and investment recommendations
- Location-based statistics and comparisons

### 3. **Conversational Interface**
- Context-aware responses using Gemini AI
- Multi-turn conversations with memory
- Intent classification for optimal routing

### 4. **Extensible Architecture**
- Modular design for easy integration with future features
- Shared Gemini service across all components
- Centralized configuration management

## 🚀 Quick Start

### Installation
```bash
# Install required packages
pip install google-generativeai pydantic xgboost lightgbm

# Clone the repository
git clone https://github.com/yourusername/RealyticsAI-github.git
cd RealyticsAI-github
```

### Running the Chatbot
```bash
# Simple demo chatbot
python demo_chatbot.py

# Full-featured chatbot (with all modules)
python backend/services/chatbot_orchestrator/realytics_chatbot.py
```

## 📁 Project Structure

```
RealyticsAI-github/
├── backend/
│   ├── core/
│   │   └── config.py                    # Centralized configuration with API keys
│   ├── services/
│   │   ├── gemini_service.py           # Gemini API service module
│   │   ├── price_prediction/
│   │   │   ├── nlp_interface.py        # NLP interface for price prediction
│   │   │   ├── model_building_advanced.py
│   │   │   ├── feature_engineering_advanced.py
│   │   │   ├── hyperparameter_tuning.py
│   │   │   └── train_enhanced_model.py
│   │   └── chatbot_orchestrator/
│   │       └── realytics_chatbot.py    # Main chatbot orchestrator
├── data/
│   └── models/                         # Trained ML models
├── demo_chatbot.py                     # Simple demo implementation
└── test_gemini_integration.py          # Integration tests
```

## 🔑 Configuration

The system uses a centralized configuration in `backend/core/config.py`:

```python
# API Configuration
GEMINI_API_KEY = "AIzaSyBS5TJCebmoyy9QyE_R-OAaYYV9V2oM-A8"  # Your API key
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 2048

# Feature Flags
ENABLE_PRICE_PREDICTION = True
ENABLE_PROPERTY_RECOMMENDATION = False  # Coming soon
ENABLE_NEGOTIATION_AGENT = False       # Coming soon
```

## 💬 Example Queries

### Price Prediction
```
User: What's the price of a 3BHK apartment in Whitefield with 1500 sqft?
Bot: Based on my analysis, a 3BHK apartment in Whitefield with 1500 sqft 
     is estimated at ₹95-110 Lakhs. This is slightly above the area average...
```

### Market Analysis
```
User: Which areas in Bengaluru are best for investment?
Bot: Based on current market trends, the top investment areas are:
     1. Whitefield - High IT presence, appreciation rate 8-10% annually
     2. Electronic City - Growing infrastructure, affordable prices...
```

### General Queries
```
User: What factors affect property prices in Bengaluru?
Bot: Key factors influencing Bengaluru property prices include:
     1. Location and proximity to IT hubs
     2. Infrastructure development
     3. Property size and configuration...
```

## 🏗️ Architecture

### 1. **Gemini Service Layer**
- `GeminiService`: Core API wrapper for Gemini
- `GeminiChatbot`: High-level chatbot interface
- Handles context management and conversation history

### 2. **NLP Interface Layer**
- `PricePredictionNLP`: Natural language processing for price queries
- Feature extraction from text
- Response generation with market context

### 3. **Orchestration Layer**
- `RealyticsAIChatbot`: Main orchestrator
- Intent classification and routing
- Feature module integration

### 4. **ML Pipeline**
- Advanced feature engineering (50+ features)
- Ensemble models (XGBoost, LightGBM, Random Forest)
- 99.6% accuracy (R² = 0.9957)

## 📊 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Voting Ensemble** | 0.9957 | 3.50 | 1.10 |
| Stacking Ensemble | 0.9956 | 3.57 | 1.22 |
| LightGBM | 0.9950 | 3.77 | 1.63 |
| XGBoost | 0.9950 | 3.79 | 1.36 |

## 🔄 Future Enhancements

### Property Recommendation (Coming Soon)
- Personalized property suggestions
- Multi-criteria filtering
- Similar property matching

### Negotiation Agent (Coming Soon)
- Automated price negotiation strategies
- Market-based bargaining insights
- Deal optimization recommendations

## 🛠️ Development

### Adding New Features
1. Create feature module in `backend/services/`
2. Add NLP interface for natural language processing
3. Register in chatbot orchestrator
4. Update intent classification

### Testing
```bash
# Run integration tests
python test_gemini_integration.py

# Test specific modules
python -m pytest tests/
```

## 📝 API Usage

### Python Integration
```python
from backend.services.gemini_service import get_gemini_chatbot
from backend.services.price_prediction.nlp_interface import create_nlp_interface

# Initialize chatbot
chatbot = get_gemini_chatbot()

# Process query
response = chatbot.process_message("What's the price of a 2BHK in Koramangala?")
print(response)
```

### REST API (Coming Soon)
```python
POST /api/chat
{
    "message": "What's the average price in Whitefield?",
    "context": {"session_id": "12345"}
}
```

## 🔒 Security Notes

- API key is currently hardcoded for demo purposes
- In production, use environment variables or secure key management
- Implement rate limiting for API calls
- Add user authentication for personalized features

## 📚 Dependencies

```
google-generativeai>=0.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
joblib>=1.3.0
rich>=13.0.0
pydantic>=2.0.0
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is part of RealyticsAI and follows the project's licensing terms.

## 🙏 Acknowledgments

- Google Gemini API for natural language processing
- Bengaluru housing dataset for training data
- Open-source ML libraries for model development

---

**Note**: The Gemini API key provided is for demonstration purposes. Please use your own API key for production deployments.

For questions or support, please open an issue in the repository.
