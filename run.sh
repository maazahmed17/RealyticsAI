#!/bin/bash

# RealyticsAI Launcher Script

echo "=================================================================================="
echo "                 🏠 RealyticsAI - Intelligent Real Estate Assistant"
echo "=================================================================================="
echo ""
echo "Starting application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

# Run the application
python3 app.py

echo ""
echo "✨ Thank you for using RealyticsAI!"
