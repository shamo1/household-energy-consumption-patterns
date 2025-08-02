# Project Structure

```
household-energy-analysis/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── install.sh                   # Installation script (Unix/Mac)
├── install.bat                  # Installation script (Windows)
├── verify_installation.py       # Installation verification
│
├── data/                        # Dataset directory
│   └── household_data_15min_singleindex.csv
│
├── notebooks/                   # Jupyter notebooks
│   └── household_energy_analysis.ipynb
│
├── src/                         # Source code
│   ├── __init__.py
│   └── energy_analysis.py       # Core analysis functions
│
├── models/                      # Trained models (to be generated)
│   ├── lstm_model.h5
│   ├── arima_model.pkl
│   └── prophet_model.pkl
│
├── results/                     # Analysis results
│   ├── model_performance.json
│   └── feature_importance.csv
│
├── visualizations/              # Generated plots and charts
│   ├── consumption_patterns.png
│   ├── seasonal_analysis.png
│   ├── model_comparison.png
│   └── forecasting_results.png
│
├── scripts/                     # Utility scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── generate_report.py
│
└── docs/                        # Additional documentation
    ├── methodology.md
    ├── data_dictionary.md
    └── model_details.md
```

## Directory Descriptions

- **data/**: Contains the raw and processed datasets
- **notebooks/**: Jupyter notebooks for interactive analysis
- **src/**: Core Python modules and functions
- **models/**: Saved machine learning models
- **results/**: Analysis outputs and metrics
- **visualizations/**: Generated plots and figures
- **scripts/**: Standalone utility scripts
- **docs/**: Detailed documentation and methodology
