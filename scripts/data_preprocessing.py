"""
Data Preprocessing Module

This module contains functions for cleaning and preprocessing
the household energy consumption dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(filepath):
    """Load the household energy dataset."""
    return pd.read_csv(filepath)


def clean_missing_values(df, method='interpolate'):
    """Clean missing values using specified method."""
    if method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'forward_fill':
        return df.ffill()
    elif method == 'drop':
        return df.dropna()
    return df


def detect_outliers(df, method='iqr'):
    """Detect outliers using IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ~((df < lower_bound) | (df > upper_bound)).any(axis=1)


def scale_features(df, method='standard'):
    """Scale features using specified method."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    
    return scaler.fit_transform(df), scaler
