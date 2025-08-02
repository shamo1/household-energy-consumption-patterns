@echo off
REM Household Energy Consumption Analysis - Windows Installation Script
REM This script installs all required dependencies for the analysis notebook

echo 🏠 Household Energy Consumption Analysis - Setup Script
echo ========================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found
pip --version

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install virtualenv if not present
echo 🔧 Checking for virtualenv...
pip show virtualenv >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing virtualenv...
    pip install virtualenv
)

REM Create virtual environment
echo 🌱 Creating virtual environment 'energy_analysis_env'...
python -m virtualenv energy_analysis_env

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call energy_analysis_env\Scripts\activate.bat

REM Install requirements
echo 📦 Installing required packages...
pip install -r requirements.txt

echo.
echo ✅ Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Activate the virtual environment: energy_analysis_env\Scripts\activate.bat
echo 2. Start Jupyter: jupyter notebook
echo 3. Open the notebook: 2212172.ipynb
echo.
echo 📊 Your environment is ready for household energy consumption analysis!
echo.
echo 🔍 To verify installation, run: python -c "import pandas, numpy, sklearn, tensorflow, prophet; print('All packages imported successfully!')"
echo.
pause
