#!/bin/bash

# Streamlit Web App Startup Script
# Run the Flood Risk Prediction Streamlit Application

echo "========================================"
echo "🌊 Flood Risk Prediction - Web App"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv .venv"
    echo "Then: source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit
fi

echo "✅ Virtual environment activated"
echo ""
echo "🚀 Starting Streamlit Web Application..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit app
streamlit run app.py --logger.level=info
