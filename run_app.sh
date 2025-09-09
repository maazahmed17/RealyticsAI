#!/bin/bash

# RealyticsAI Startup Script
# ==========================

echo "=========================================="
echo "🏠 RealyticsAI - Starting Application"
echo "=========================================="
echo ""

# Check if Python dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask..."
    pip install flask flask-cors
fi

# Set environment variable for Gemini API if needed
export GEMINI_API_KEY=${GEMINI_API_KEY:-"your-api-key-here"}

# Start the server
echo ""
echo "🚀 Starting RealyticsAI server..."
echo "=========================================="
echo ""
echo "📱 Open your browser and go to:"
echo "   http://localhost:5000"
echo ""
echo "   Or if accessing from another device:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run the Flask application
python3 api_server.py
