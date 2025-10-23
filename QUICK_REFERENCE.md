# RealyticsAI - Quick Reference Card

## 🚀 Start the System

```bash
cd /home/maaz/RealyticsAI
python run_unified_system.py
```

**Choose:**
- **1** = Terminal Chat (Interactive)
- **2** = Web Interface (http://localhost:5000)
- **3** = Run Tests
- **4** = Help

---

## 💬 Sample Queries

### Price Predictions
```
"What's the price of a 3 BHK apartment in Whitefield?"
"Price estimate for 1500 sqft house in Koramangala"
"How much is a 2 BHK with 2 bathrooms in HSR Layout?"
```

### Property Search
```
"Find me 2 BHK apartments under 50 lakhs"
"Show properties in Electronic City with 1500+ sqft"
"Recommend furnished flats in Whitefield"
```

### Market Analysis
```
"What are the market trends in Indiranagar?"
"Which areas are best for investment?"
"Compare Whitefield vs Electronic City prices"
```

---

## 📊 Current Model Stats

- **Model:** XGBoost with Advanced Feature Engineering
- **Accuracy:** R² = 0.9959 (99.59%)
- **Error:** RMSE = 2.89 Lakhs, MAE = 0.55 Lakhs
- **Dataset:** 150,000 Bengaluru properties
- **Features:** 28 engineered features
- **Model File:** `xgboost_advanced_20251023_115658.pkl`

---

## 🔧 Useful Commands

### Retrain Model
```bash
cd /home/maaz/RealyticsAI/backend/services/price_prediction
python train_xgboost_advanced.py
```

### Check Model Metrics
```bash
cat /home/maaz/RealyticsAI/data/models/metrics_20251023_115658.json
```

### Test Web API
```bash
# Health check
curl http://localhost:5000/api/system/health

# Chat
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Price of 3 BHK in Koramangala?"}'
```

---

## 📚 Documentation

- `PROJECT_COMPLETION_SUMMARY.md` - Full project details
- `SYSTEM_ANALYSIS_REPORT.md` - Technical analysis
- `QUICK_START_AFTER_FIXES.md` - Setup guide
- `QUICK_REFERENCE.md` - This file

---

## ✅ System Status

All systems operational:
- ✅ Model: XGBoost (R² = 0.9959)
- ✅ Dataset: 150K properties loaded
- ✅ Recommendations: Real-time search
- ✅ Terminal Chat: Working
- ✅ Web API: All endpoints active
- ✅ Gemini AI: Routing functional

---

**Status:** 🟢 PRODUCTION READY
