#!/bin/bash

# Household Energy Consumption Analysis - Installation Script
# This script installs all required dependencies for the analysis notebook

echo "🏠 Household Energy Consumption Analysis - Setup Script"
echo "========================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install virtualenv if not present
echo "🔧 Checking for virtualenv..."
if ! python3 -m pip show virtualenv &> /dev/null; then
    echo "📦 Installing virtualenv..."
    python3 -m pip install virtualenv
fi

# Create virtual environment
echo "🌱 Creating virtual environment 'energy_analysis_env'..."
python3 -m virtualenv energy_analysis_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source energy_analysis_env/bin/activate

# Install requirements
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Additional installations for specific platforms
echo "🔧 Installing additional platform-specific packages..."

# For Mac users with Apple Silicon
if [[ $(uname -m) == "arm64" && $(uname -s) == "Darwin" ]]; then
    echo "🍎 Detected Apple Silicon Mac - installing optimized packages..."
    pip install tensorflow-macos tensorflow-metal
fi

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source energy_analysis_env/bin/activate"
echo "2. Start Jupyter: jupyter notebook"
echo "3. Open the notebook: 2212172.ipynb"
echo ""
echo "📊 Your environment is ready for household energy consumption analysis!"
echo ""
echo "🔍 To verify installation, run: python -c 'import pandas, numpy, sklearn, tensorflow, prophet; print(\"All packages imported successfully!\")'"
