# 🏠⚡ Household Energy Consumption Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project analyzing household energy consumption patterns using time series forecasting, deep learning, and statistical modeling techniques.

## 🚀 Quick Start Guide

### 📋 Prerequisites

- Python 3.8+ installed on your system
- 8GB+ RAM (recommended)
- 2GB free disk space

### ⚡ One-Command Setup

**For macOS/Linux:**

```bash
git clone https://github.com/shamo1/household-energy-consumption-patterns.git
cd household-energy-consumption-patterns
chmod +x install.sh && ./install.sh
```

**For Windows:**

```bash
git clone https://github.com/shamo1/household-energy-consumption-patterns.git
cd household-energy-consumption-patterns
install.bat
```

### 🔧 Manual Setup (Alternative)

1. **Clone and navigate to project:**

```bash
git clone https://github.com/shamo1/household-energy-consumption-patterns.git
cd household-energy-consumption-patterns
```

2. **Create Python virtual environment:**

```bash
python -m venv .venv
```

3. **Activate environment:**

```bash
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

5. **Verify installation:**

```bash
python verify_installation.py
```

### 🎯 Running the Project

#### Option 1: Interactive Analysis (Recommended)

**Open the complete analysis notebook:**

```bash
jupyter notebook notebooks/household_energy_analysis.ipynb
```

> **📝 Important**: The notebook contains all the code and runs smoothly with complete analysis, visualizations, and results. If any scripts don't run properly, refer to the notebook which has all working code.

#### Option 2: Automated Scripts

**Generate all visualizations:**

```bash
python scripts/generate_visualizations.py
```

**Run model training:**

```bash
python scripts/model_training.py
```

**Run complete analysis:**

```bash
python src/energy_analysis.py
```

## 🎯 Project Objectives

- **Pattern Recognition**: Identify consumption patterns across different appliances and time periods
- **Predictive Modeling**: Forecast future energy consumption using multiple ML approaches
- **Seasonal Analysis**: Understand seasonal variations and their impact on energy usage
- **Energy Optimization**: Provide insights for efficient energy management

## 📊 Dataset Overview

The analysis uses a comprehensive household energy dataset with:

- **Timespan**: 2015-2018 (4 years)
- **Frequency**: 15-minute intervals
- **Total Records**: 210,240 data points
- **Variables**: 9 features including 7 household appliances

### 🔌 Monitored Appliances

- Dishwasher
- Electric Vehicle
- Freezer
- Heat Pump
- Photovoltaic (PV) System
- Refrigerator
- Washing Machine

## 🧠 Machine Learning Models & Results

### Model Performance Comparison

| Model             | R² Score   | RMSE       | MAE        | Use Case              |
| ----------------- | ---------- | ---------- | ---------- | --------------------- |
| **LSTM**          | **0.8921** | **0.1158** | **0.0823** | Real-time forecasting |
| Prophet           | 0.8856     | 0.1192     | 0.0845     | Business planning     |
| Linear Regression | 0.8743     | 0.1248     | 0.0892     | Baseline comparison   |
| ARIMA             | 0.8567     | 0.1334     | 0.0967     | Statistical analysis  |

### Key Findings

#### 🌟 **Energy Consumption Patterns**

- **Peak Hours**: 6 PM - 9 PM (evening routine)
- **Low Usage**: 2 AM - 6 AM (overnight period)
- **Weekend Pattern**: Different from weekdays - clear behavioral differences
- **Seasonal Impact**: 23% higher consumption in winter months

#### 🔋 **Renewable Energy Integration**

- **Solar Generation**: Peak output 12 PM - 3 PM
- **Net Energy**: 15% average export during summer
- **Grid Independence**: 40% self-consumption rate

## 📁 Project Structure

```
household-energy-analysis/
├── 📊 data/                     # Datasets
│   └── household_data_15min_singleindex.csv
├── 📓 notebooks/                # Jupyter notebooks
│   └── household_energy_analysis.ipynb  # MAIN ANALYSIS
├── 🐍 src/                      # Source code
│   └── energy_analysis.py
├── 🤖 models/                   # Trained models
├── 📈 results/                  # Analysis outputs
│   ├── model_performance.json
│   └── feature_importance.csv
├── 🎨 visualizations/           # Generated plots
├── 🔧 scripts/                  # Utility scripts
│   ├── generate_visualizations.py
│   ├── model_training.py
│   └── data_preprocessing.py
└── 📚 docs/                     # Documentation
```

## 🔧 Troubleshooting

### If Scripts Don't Work:

**Use the Jupyter Notebook** - `notebooks/household_energy_analysis.ipynb`

The notebook contains:

- ✅ All working code
- ✅ Complete analysis pipeline
- ✅ All visualizations
- ✅ Smooth execution
- ✅ Detailed explanations

### Common Issues:

1. **Environment Issues**: Make sure virtual environment is activated
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Data Path Issues**: Ensure data files are in the correct location
4. **Python Version**: Use Python 3.8 or higher

### Verification Commands:

```bash
# Check Python version
python --version

# Check if packages are installed
python -c "import pandas, numpy, sklearn, tensorflow; print('All packages installed')"

# Verify data file exists
ls -la data/household_data_15min_singleindex.csv
```

## 📈 Generated Outputs

After running the analysis, you'll find:

### Visualizations (`visualizations/` folder):

- **EDA Charts**: Data exploration and patterns
- **Model Performance**: Training curves and predictions
- **Energy Patterns**: Weekday vs weekend analysis
- **Seasonal Analysis**: Consumption by seasons
- **Feature Analysis**: Correlation heatmaps

### Results (`results/` folder):

- **model_performance.json**: Performance metrics for all models
- **feature_importance.csv**: Feature ranking and importance scores

### Models (`models/` folder):

- Trained machine learning models
- Model configurations and parameters

## 💡 Key Insights from Analysis

1. **LSTM Neural Network** achieved the best performance (R² = 0.8921)
2. **Weekdays vs Weekends** show clear behavioral differences in energy usage
3. **Heat Pump** is the primary consumption driver (34.2% importance)
4. **Evening hours (6-9 PM)** represent peak consumption periods
5. **Winter consumption** is 23% higher than summer
6. **Energy optimization potential** of 15-20% identified

## 📞 Support

If you encounter any issues:

1. **First**: Try the Jupyter notebook - it has all working code
2. **Check**: Virtual environment is activated
3. **Verify**: All dependencies are installed
4. **Review**: The troubleshooting section above

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
