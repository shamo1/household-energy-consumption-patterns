# 📊 Visualizations Directory

This directory contains all generated visualizations from the household energy analysis.

## 🎯 Quick Start

Generate all visualizations with:

```bash
python scripts/generate_visualizations.py
```

## 📁 Directory Structure

```
visualizations/
├── eda/                          # Exploratory Data Analysis
│   ├── missing_values_heatmap.png      # Data quality overview
│   ├── consumption_distribution.png    # Appliance usage patterns
│   ├── hourly_consumption_pattern.png  # Daily usage cycles
│   ├── seasonal_consumption.png        # Seasonal variations
│   └── weekday_vs_weekend.png         # Behavioral differences
├── features/                     # Feature Analysis
│   ├── correlation_heatmap.png         # Appliance correlations
│   └── target_correlation.png          # Predictive features
├── forecasting/                  # Time Series Analysis
│   ├── time_series_overview.png        # Long-term trends
│   └── monthly_consumption.png         # Monthly patterns
├── models/                       # Model Performance
│   ├── lstm_training_loss.png          # Deep learning progress
│   ├── prediction_accuracy.png         # Model validation
│   └── error_distribution.png          # Error analysis
├── summary_dashboard.png         # Executive overview
└── project_overview.png          # Main project visualization
```

## 🔧 Generation Options

### Generate Specific Categories

```bash
# Exploratory Data Analysis only
python scripts/generate_visualizations.py --type eda

# Feature analysis only
python scripts/generate_visualizations.py --type features

# Time series forecasting only
python scripts/generate_visualizations.py --type forecasting

# Model performance only
python scripts/generate_visualizations.py --type models
```

### Custom Output

```bash
# Custom output directory
python scripts/generate_visualizations.py --output-dir custom_charts/

# Specific data file
python scripts/generate_visualizations.py --data-path path/to/data.csv
```

## 📈 Visualization Types

### 🔍 **Exploratory Data Analysis (EDA)**

- **Purpose**: Understand data quality and patterns
- **Key Insights**: Missing data, outliers, distributions
- **Business Value**: Data reliability assessment

### 🔗 **Feature Analysis**

- **Purpose**: Identify important predictors
- **Key Insights**: Appliance correlations, feature importance
- **Business Value**: Model optimization guidance

### ⏱️ **Time Series Analysis**

- **Purpose**: Understand temporal patterns
- **Key Insights**: Trends, seasonality, cycles
- **Business Value**: Forecasting foundation

### 🤖 **Model Performance**

- **Purpose**: Validate prediction accuracy
- **Key Insights**: Model comparison, error analysis
- **Business Value**: Deployment readiness

## 🎨 Visualization Standards

All visualizations follow these standards:

- **High Resolution**: 300 DPI for presentations
- **Consistent Colors**: Professional color palette
- **Clear Labels**: Descriptive titles and axes
- **Statistical Context**: Annotations with key metrics
- **Business Focus**: Actionable insights highlighted

## 📊 Usage in Documentation

These visualizations are referenced in:

- `README.md` - Project overview and methodology
- `docs/methodology.md` - Technical analysis details
- `PROJECT_SUMMARY.md` - Executive summary
- Jupyter notebooks - Interactive analysis

## 🔄 Regeneration

To update visualizations after data changes:

```bash
# Clean and regenerate all
rm -rf visualizations/
python scripts/generate_visualizations.py

# Or use the force flag (when implemented)
python scripts/generate_visualizations.py --force
```

## 💡 Tips

1. **Performance**: Large datasets may take 2-5 minutes to process
2. **Memory**: Ensure 4GB+ RAM available for complex visualizations
3. **Display**: Best viewed on monitors with 1920x1080+ resolution
4. **Sharing**: All images are optimized for web and print use

---

**Note**: Run `python scripts/generate_visualizations.py` to populate this directory with actual visualizations.
