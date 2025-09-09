# 🏠 RealyticsAI - Intelligent Real Estate Assistant

An AI-powered real estate assistant for Bengaluru property market, featuring advanced ML models and natural language processing through Google's Gemini API.

## ✨ Features

- **Property Price Prediction** - Get instant property valuations with 99.57% accuracy
- **Natural Language Interface** - Chat naturally using Gemini 1.5 Flash
- **Market Analysis** - Data-driven insights from 13,320+ properties
- **Smart Recommendations** - Location-based and budget-based suggestions

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RealyticsAI-github.git
cd RealyticsAI-github
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## 💬 Usage Examples

Simply describe your property to get instant price estimates:

```
You: 3 BHK, 1500 sqft, 2 bathrooms in Whitefield
Bot: Estimated Price: ₹125.50 Lakhs (Range: ₹110-140 Lakhs)

You: 2 BHK apartment in Koramangala, 1000 sqft
Bot: Estimated Price: ₹95.75 Lakhs (Range: ₹85-105 Lakhs)
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.9957 |
| RMSE | 3.50 Lakhs |
| MAE | 1.10 Lakhs |
| Dataset | 13,320 properties |

## 🏗️ Project Structure

```
RealyticsAI-github/
├── app.py                 # Main application entry point
├── src/
│   └── chatbot.py        # Chatbot implementation
├── config/
│   └── settings.py       # Configuration settings
├── data/
│   └── models/           # Trained ML models
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Gemini API key
- Data paths
- Model settings

## 📝 License

This project is part of the RealyticsAI system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.
