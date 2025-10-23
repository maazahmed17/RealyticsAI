#!/bin/bash
# Cleanup Script for RealyticsAI
# Removes unnecessary files and old models

echo "==============================================="
echo "RealyticsAI Project Cleanup"
echo "==============================================="

# Remove old overfit model
echo "🗑️  Removing old overfit model..."
rm -f /home/maaz/RealyticsAI/data/models/xgboost_advanced_*.pkl

# Remove duplicate/unnecessary files in backend
echo "🗑️  Removing duplicate ML code in backend..."
rm -f /home/maaz/RealyticsAI/backend/services/price_prediction/train_*.py
rm -f /home/maaz/RealyticsAI/backend/services/price_prediction/hyperparameter_tuning.py

# Remove old training output
echo "🗑️  Removing old training logs..."
rm -f /home/maaz/RealyticsAI/src/training_output.log

# Remove __pycache__ directories
echo "🗑️  Removing __pycache__ directories..."
find /home/maaz/RealyticsAI -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
echo "🗑️  Removing .pyc files..."
find /home/maaz/RealyticsAI -type f -name "*.pyc" -delete

# Remove old model metrics
echo "🗑️  Removing old model metrics..."
rm -f /home/maaz/RealyticsAI/data/models/metrics_*.json
rm -f /home/maaz/RealyticsAI/data/models/model_metrics.json

# List remaining models
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Remaining models:"
ls -lh /home/maaz/RealyticsAI/data/models/*.pkl 2>/dev/null | tail -5

echo ""
echo "💾 Current disk usage of models directory:"
du -sh /home/maaz/RealyticsAI/data/models/

echo ""
echo "✅ Use the FIXED model for production:"
echo "   xgboost_fixed_*.pkl"
echo ""
