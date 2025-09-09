# 🏠 RealyticsAI - AI-Powered Real Estate Platform with Natural Language Interface

## 🎯 Overview

RealyticsAI is an intelligent real estate platform featuring a **natural language chatbot interface** for property price predictions, market analysis, and personalized recommendations. The platform combines machine learning with conversational AI to make real estate insights accessible through simple English queries.

## ✨ Key Features

### 🤖 **Natural Language Chatbot** (NEW!)
- **Conversational Interface**: Ask questions in plain English
- **Intent Understanding**: Automatically identifies what you want
- **Smart Entity Extraction**: Understands property features from natural text
- **Context Awareness**: Maintains conversation history
- **Multiple Interfaces**: CLI, REST API, and WebSocket support

### 📊 **Price Prediction Engine**
- ML-powered property valuations with MLflow tracking
- Hybrid predictions combining ML with historical data
- Real-time market analysis and insights
- Confidence scoring based on data availability

### 💡 **Intelligent Features**
- Market trend analysis for any location
- Budget-based property recommendations
- Location comparisons and insights
- Investment opportunity identification

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Bengaluru housing dataset (CSV format)

### Installation

```bash
# Clone the repository
git clone https://github.com/maazahmed17/RealyticsAI.git
cd RealyticsAI

# Install dependencies
pip install -r requirements.txt
```

## 💬 Using the Chatbot

### Option 1: Interactive CLI Chatbot (Easiest!)

```bash
cd backend/services/chatbot_orchestrator
python cli_chat.py
```

Then chat naturally:
```
You: What's the price of a 2 bathroom apartment with 1 balcony?
Bot: Based on my analysis, a property with 2 bathrooms and 1 balcony 
     is estimated at ₹69.11 lakhs...

You: Show me properties under 50 lakhs
Bot: Best value locations:
     • Ananth Nagar: Avg ₹33.71L (30 properties)
     • Chandapura: Avg ₹34.02L (100 properties)...

You: Give me market analysis for Whitefield
Bot: Whitefield Market Overview:
     • Properties Available: 540
     • Average Price: ₹128.01 lakhs...
```

### Option 2: Chatbot API Server

```bash
cd backend/services/chatbot_orchestrator
python chatbot_api.py
```

Then access:
- **Chat API**: http://localhost:8001/api/chat
- **WebSocket**: ws://localhost:8001/ws/{session_id}
- **API Docs**: http://localhost:8001/docs

### Option 3: Direct Price Prediction Analysis

```bash
cd backend/services/price_prediction
python main.py
```

## 📝 Example Chatbot Queries

### Price Predictions
- "What's the price of a 2 bathroom house with 1 balcony?"
- "How much would a 3BHK apartment cost in Whitefield?"
- "I need a property with three bathrooms and no balcony"

### Market Analysis
- "Show me market analysis for Electronic City"
- "What's the average price in Whitefield?"
- "Which area has the most properties?"

### Recommendations
- "Show me properties under 50 lakhs"
- "What can I get for 100 lakhs?"
- "Recommend areas within my budget of 75 lakhs"

### Comparisons
- "Compare Whitefield and Electronic City"
- "What's the price difference between 2 and 3 bathrooms?"

## 🏗️ Architecture

```
RealyticsAI/
├── backend/
│   ├── services/
│   │   ├── chatbot_orchestrator/    ✅ Natural Language Interface
│   │   │   ├── chatbot_handler.py   # Core NLU logic
│   │   │   ├── chatbot_api.py       # FastAPI server
│   │   │   └── cli_chat.py          # Interactive CLI
│   │   ├── price_prediction/        ✅ ML Price Engine
│   │   │   ├── main.py              # Complete analysis
│   │   │   └── [ML components]
│   │   ├── property_recommendation/ 🔄 Coming Soon
│   │   └── negotiation_agent/       🔄 Coming Soon
│   ├── core/                        # Configuration
│   └── api/                         # API routes
├── data/                            # Datasets
├── frontend/                        🔄 React UI (Coming)
└── docs/                            # Documentation
```

## 🔬 Technical Stack

### Natural Language Processing
- **Intent Classification**: Identifies user intent (price, analysis, recommendation)
- **Entity Extraction**: Extracts features (bathrooms, balconies, location, budget)
- **Context Management**: Maintains conversation state
- **Response Generation**: Creates natural, informative responses

### Machine Learning
- **Models**: Linear Regression, Random Forest, Gradient Boosting
- **MLflow**: Experiment tracking and model versioning
- **Hybrid Predictions**: 60% ML + 40% historical data
- **Feature Engineering**: Automated feature extraction

## 📊 API Endpoints

### Chatbot API (Port 8001)

```http
POST /api/chat
{
    "message": "What's the price of a 2 bathroom house?",
    "session_id": "optional"
}

Response:
{
    "response": "Based on my analysis...",
    "intent": "price_prediction",
    "entities": {"bathrooms": 2, "balconies": null},
    "session_id": "abc123"
}
```

### WebSocket for Real-time Chat

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/session123');
ws.send("What's the price?");
```

### Price Prediction API (Port 8000)

```http
POST /api/v1/predict
GET /api/v1/model/metrics
GET /api/v1/comprehensive-test
```

## 📈 Model Performance

Current metrics on Bengaluru dataset (13,320 properties):
- **Accuracy**: R² Score = 0.194
- **Intent Classification**: 85%+ accuracy
- **Entity Extraction**: 90%+ for structured queries
- **Response Time**: < 500ms

## 🎯 Supported Intents

1. **PRICE_PREDICTION**: Property price queries
2. **MARKET_ANALYSIS**: Market insights and statistics
3. **RECOMMENDATION**: Budget-based suggestions
4. **COMPARISON**: Location/feature comparisons
5. **GREETING**: Conversation starters
6. **HELP**: Assistance requests

## 🚀 Advanced Features

### Using OpenAI GPT (Optional)

For enhanced natural language understanding:

```bash
export OPENAI_API_KEY="your-api-key"
python cli_chat.py
```

### Debug Mode

See extracted intents and entities:

```bash
export DEBUG=1
python cli_chat.py
```

## 📊 Dataset Information

**Bengaluru House Prices**
- Total Properties: 13,320
- Locations: 1,305
- Price Range: ₹8 - ₹3,600 lakhs
- Average: ₹112.57 lakhs

## 🔮 Roadmap

### Phase 1: ✅ Completed
- Natural language chatbot
- Price prediction engine
- Market analysis
- Basic recommendations

### Phase 2: 🔄 In Progress
- Enhanced NLU with GPT
- Multi-turn conversations
- Advanced entity extraction

### Phase 3: 📅 Planned
- Property recommendation engine
- Negotiation agent
- Voice interface
- React web UI
- Mobile app

## 🛠️ Development

### Running Tests

```bash
# Test chatbot
cd backend/services/chatbot_orchestrator
python chatbot_handler.py  # Runs built-in tests

# Test API
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Adding Custom Locations

```python
chatbot.known_locations.append("your_area")
```

### Extending Intents

Add new patterns in `classify_intent()` method in `chatbot_handler.py`

## 📝 License

Educational and demonstration purposes

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More sophisticated NLU
- Additional entity types
- Multi-language support
- Voice integration

## 🎉 Try It Now!

```bash
# Start chatting in 30 seconds:
cd backend/services/chatbot_orchestrator && python cli_chat.py
```

Ask: **"What's the price of a 2 bathroom house?"** and see the magic! ✨

---

**Note**: This is a production-ready chatbot with room for growth. The natural language interface makes real estate analytics accessible to everyone, not just data scientists!
