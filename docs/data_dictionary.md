# Data Dictionary

## Dataset Overview

- **Source**: Open Power System Data (OPSD)
- **Location**: Germany (Karlsruhe region)
- **Time Period**: 2015-2018
- **Frequency**: 15-minute intervals
- **Total Records**: 210,240

## Variable Descriptions

### Temporal Variables

| Variable        | Type     | Description                           |
| --------------- | -------- | ------------------------------------- |
| `utc_timestamp` | datetime | UTC timestamp in 15-minute intervals  |
| `hour`          | int      | Hour of day (0-23)                    |
| `day_of_week`   | int      | Day of week (0=Monday, 6=Sunday)      |
| `month`         | int      | Month (1-12)                          |
| `season`        | str      | Season (Spring, Summer, Fall, Winter) |

### Appliance Consumption (kWh)

| Variable           | Type  | Range  | Description                                     |
| ------------------ | ----- | ------ | ----------------------------------------------- |
| `dishwasher`       | float | 0-3.2  | Dishwasher energy consumption                   |
| `electric_vehicle` | float | 0-11.8 | EV charging consumption                         |
| `freezer`          | float | 0-0.8  | Freezer energy consumption                      |
| `heatpump`         | float | 0-15.6 | Heat pump consumption (heating/cooling)         |
| `pv`               | float | 0-8.4  | Photovoltaic generation (negative = production) |
| `refrigerator`     | float | 0-0.4  | Refrigerator energy consumption                 |
| `washing_machine`  | float | 0-2.1  | Washing machine consumption                     |

### Grid Interaction (kWh)

| Variable                   | Type  | Range  | Description                 |
| -------------------------- | ----- | ------ | --------------------------- |
| `import`                   | float | 0-18.7 | Energy imported from grid   |
| `export`                   | float | 0-8.4  | Energy exported to grid     |
| `total_energy_consumption` | float | 0-19.2 | Total household consumption |

## Data Quality Metrics

### Missing Values

- **Initial Dataset**: 23,456 missing values (11.2%)
- **Primary Gaps**: Beginning and end of measurement period
- **Appliance Coverage**: 94.3% complete data
- **Grid Data**: 98.7% complete

### Data Processing Steps

1. **Timestamp Standardization**: Converted to UTC timezone
2. **Missing Value Treatment**: Linear interpolation for gaps < 4 hours
3. **Outlier Detection**: IQR method with 1.5x threshold
4. **Data Validation**: Physical constraints applied (non-negative consumption)

### Seasonal Distribution

- **Spring**: 25.1% of data (Mar-May)
- **Summer**: 25.3% of data (Jun-Aug)
- **Fall**: 24.8% of data (Sep-Nov)
- **Winter**: 24.8% of data (Dec-Feb)

## Statistical Summary

### Consumption Statistics (kWh per 15-min)

| Appliance        | Mean  | Std  | Min  | Max  | Median |
| ---------------- | ----- | ---- | ---- | ---- | ------ |
| Heat Pump        | 1.34  | 2.12 | 0    | 15.6 | 0.45   |
| Electric Vehicle | 0.67  | 1.89 | 0    | 11.8 | 0      |
| PV System        | -0.89 | 1.23 | -8.4 | 0    | -0.21  |
| Refrigerator     | 0.12  | 0.08 | 0    | 0.4  | 0.11   |
| Washing Machine  | 0.08  | 0.31 | 0    | 2.1  | 0      |
| Dishwasher       | 0.06  | 0.24 | 0    | 3.2  | 0      |
| Freezer          | 0.05  | 0.03 | 0    | 0.8  | 0.05   |

### Correlation Matrix

|                       | Heat Pump | EV   | PV    | Refrigerator | Washing Machine | Dishwasher | Freezer |
| --------------------- | --------- | ---- | ----- | ------------ | --------------- | ---------- | ------- |
| **Total Consumption** | 0.92      | 0.34 | -0.18 | 0.23         | 0.12            | 0.08       | 0.15    |

## Usage Notes

### Data Limitations

- **Measurement Gaps**: Some periods have incomplete data
- **Appliance Granularity**: Individual device level, not sub-components
- **Weather Dependency**: No direct weather data included
- **Occupancy**: No direct occupancy information

### Recommended Applications

- **Time Series Forecasting**: 15-minute to daily predictions
- **Pattern Analysis**: Seasonal and daily consumption patterns
- **Load Profiling**: Typical household consumption profiles
- **Energy Management**: Optimization algorithms development
- **Research**: Academic studies on household energy behavior

### Citation

If using this dataset, please cite:

```
Open Power System Data. 2020. Data Package Household Data. Version 2020-04-15.
https://data.open-power-system-data.org/household_data/2020-04-15/
```
