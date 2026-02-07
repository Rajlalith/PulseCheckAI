#!/bin/bash
# PulseCheck AI - Quick Setup Script
# This script sets up PulseCheck AI for development

set -e

echo "🚀 PulseCheck AI - Setup Script"
echo "================================"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Upgrade pip
echo "📤 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env with your Twitter API token"
fi

# Create logs directory
mkdir -p logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env with your Twitter Bearer Token (optional for demo)"
echo "2. Run: streamlit run app.py"
echo ""
echo "Useful commands:"
echo "- Activate environment: source venv/bin/activate"
echo "- Run tests: pytest tests/ -v"
echo "- Format code: black src/ tests/ app.py"
echo "- Run linting: flake8 src/ tests/ app.py"
echo ""
