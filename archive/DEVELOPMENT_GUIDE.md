# 🛠️ RealyticsAI Development Guide

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose (optional)

### **1. Setup Backend**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### **2. Setup Frontend**
```bash
cd frontend
npm install
npm start
```

### **3. Docker Setup (Alternative)**
```bash
docker-compose up --build
```

---

## 🏗️ **Project Structure Overview**

```
realyticsAI/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Main application entry
│   ├── api/routes/            # API endpoints
│   ├── services/              # Business logic services
│   │   ├── price_prediction/  # ✅ Price prediction (Active)
│   │   ├── property_recommendation/  # 🚧 Coming Soon
│   │   ├── negotiation_agent/        # 🚧 Coming Soon
│   │   └── chatbot_orchestrator/     # 🚧 In Development
│   └── core/                  # Shared components
├── frontend/                  # React Frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/            # Application pages
│   │   └── services/         # API integration
├── data/                     # Data management
├── docs/                     # Documentation
└── docker/                   # Deployment configs
```

---

## 🔧 **Current Features Status**

### **✅ Active Features**
1. **Price Prediction Service**
   - REST API endpoints (`/api/v1/predict`)
   - Local data support (Bengaluru dataset)
   - Multiple file formats (CSV, Excel, ZIP)
   - Market analysis and similar properties

2. **Health Monitoring**
   - System health checks (`/api/v1/health`)
   - Performance metrics
   - Service status monitoring

3. **Basic Chatbot Interface**
   - Natural language queries (`/api/v1/chat`)
   - Intent detection (simplified)
   - Session management

### **🚧 In Development**
1. **React Frontend**
   - Modern UI with Chakra UI
   - Real-time chat interface
   - Interactive dashboards

2. **Advanced Chatbot**
   - LLM integration (planned)
   - Context-aware conversations
   - Multi-modal queries

### **📋 Planned Features**
1. **Property Recommendation Engine**
2. **Negotiation Agent AI**
3. **Advanced Analytics Dashboard**
4. **Mobile Application**

---

## 🔌 **API Endpoints**

### **Health & Status**
```http
GET /api/v1/health              # Basic health check
GET /api/v1/health/detailed     # Detailed system metrics
GET /api/v1/ping                # Simple ping
GET /api/v1/status              # Feature status
```

### **Price Prediction**
```http
POST /api/v1/predict            # Single property prediction
POST /api/v1/predict/batch      # Multiple properties
POST /api/v1/train              # Train new model
GET  /api/v1/models             # List available models
GET  /api/v1/models/{name}/status # Model details
```

### **Market Analysis**
```http
GET /api/v1/market-analysis/{location}  # Location analysis
GET /api/v1/similar-properties          # Find similar properties
```

### **Chatbot**
```http
POST /api/v1/chat                       # Chat with bot
GET  /api/v1/chat/sessions/{id}         # Chat history
DELETE /api/v1/chat/sessions/{id}       # Clear session
```

---

## 💻 **Development Workflow**

### **Adding New Features**

1. **Backend Service**
   ```bash
   # Create new service
   mkdir backend/services/your_service
   cd backend/services/your_service
   
   # Create service files
   touch __init__.py service.py models.py
   ```

2. **API Routes**
   ```bash
   # Create new route file
   touch backend/api/routes/your_feature.py
   
   # Add to main.py
   # app.include_router(your_feature.router, prefix="/api/v1", tags=["Your Feature"])
   ```

3. **Frontend Components**
   ```bash
   # Create new component
   mkdir frontend/src/components/YourComponent
   cd frontend/src/components/YourComponent
   touch YourComponent.tsx index.ts
   ```

### **Testing**
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests  
cd frontend
npm test
```

### **Code Quality**
```bash
# Backend linting
cd backend
flake8 .
black .

# Frontend linting
cd frontend
npm run lint:fix
```

---

## 🔄 **Integration Guide**

### **Adding Price Prediction to New Data**

1. **Prepare Your Data**
   - Format: CSV, Excel, or ZIP
   - Required: Target price column
   - Recommended: Property features (bedrooms, bathrooms, location, etc.)

2. **Train Model via API**
   ```http
   POST /api/v1/train
   Content-Type: multipart/form-data
   
   data_file: your_data.csv
   target_column: "price"
   model_name: "your_custom_model"
   ```

3. **Make Predictions**
   ```http
   POST /api/v1/predict
   {
     "property_features": {
       "bedrooms": 3,
       "bathrooms": 2,
       "location": "Your City"
     },
     "model_type": "your_custom_model"
   }
   ```

### **Integrating with Existing Systems**

1. **Database Integration**
   ```python
   # backend/core/database/connection.py
   from sqlalchemy import create_engine
   from core.config.settings import get_settings
   
   settings = get_settings()
   engine = create_engine(settings.database_url)
   ```

2. **External API Integration**
   ```python
   # backend/services/external/api_client.py
   import httpx
   
   async def fetch_external_data(endpoint: str):
       async with httpx.AsyncClient() as client:
           response = await client.get(endpoint)
           return response.json()
   ```

---

## 📊 **Data Management**

### **Data Flow**
```
Raw Data → Data Ingestion → Preprocessing → Model Training → Predictions
```

### **Supported Data Sources**
- **Local Files**: CSV, Excel, ZIP
- **Databases**: PostgreSQL, SQLite
- **APIs**: REST endpoints, GraphQL
- **Cloud Storage**: AWS S3, Google Cloud Storage

### **Data Storage Structure**
```
data/
├── raw/              # Original datasets
│   ├── bengaluru_house_prices.csv
│   └── your_custom_data.csv
├── processed/        # Cleaned datasets
│   ├── train_data.csv
│   └── test_data.csv
└── models/          # Trained ML models
    ├── local_prices_predictor.pkl
    └── custom_model.pkl
```

---

## 🚀 **Deployment**

### **Development Environment**
```bash
# Option 1: Direct Python/Node
python backend/main.py &
npm start --prefix frontend

# Option 2: Docker Compose
docker-compose up --build
```

### **Production Deployment**

1. **Environment Variables**
   ```bash
   # .env file
   ENVIRONMENT=production
   DEBUG=false
   DATABASE_URL=postgresql://user:pass@localhost/realyticsai
   REDIS_URL=redis://localhost:6379/0
   SECRET_KEY=your-production-secret
   ```

2. **Docker Production**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Cloud Deployment**
   - **AWS**: ECS, Lambda, RDS
   - **Google Cloud**: Cloud Run, Cloud SQL
   - **Azure**: Container Instances, Azure SQL

---

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Port Conflicts**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 PID
   ```

2. **Dependencies Issues**
   ```bash
   # Backend
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   
   # Frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Data File Path Issues**
   ```python
   # Update settings.py with correct path
   default_data_file: str = "/your/correct/path/to/data.csv"
   ```

### **Debugging**

1. **Backend Logs**
   ```bash
   # Enable debug mode
   export DEBUG=true
   python backend/main.py
   ```

2. **Frontend Debug**
   ```bash
   # Enable React debug
   export REACT_APP_DEBUG=true
   npm start
   ```

---

## 🎯 **Next Steps**

### **Immediate Tasks**
1. Complete React frontend UI
2. Integrate LLM for advanced chatbot
3. Add user authentication
4. Implement caching layer

### **Feature Roadmap**
1. **Phase 2**: Property Recommendation Engine
2. **Phase 3**: Negotiation Agent AI  
3. **Phase 4**: Mobile Application
4. **Phase 5**: Enterprise Features

### **Contributing**
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request
5. Follow code review process

---

## 📚 **Resources**

### **Documentation**
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://reactjs.org/docs/)
- [Chakra UI](https://chakra-ui.com/)
- [ZenML Docs](https://docs.zenml.io/)

### **APIs & Tools**
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Docker](https://docs.docker.com/)

This development guide will grow as the platform evolves! 🚀
