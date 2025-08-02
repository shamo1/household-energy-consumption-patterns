"""
Model Training Module

This module contains functions for training various machine learning
models for energy consumption prediction.
"""

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import joblib


def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100):
    """Train a random forest model."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_lstm_model(X_train, y_train, sequence_length=96):
    """Train an LSTM model for time series prediction."""
    model = Sequential([
        LSTM(50, input_shape=(1, sequence_length)),
        Dropout(0.4),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def save_model(model, filepath):
    """Save trained model."""
    if filepath.endswith('.pkl'):
        joblib.dump(model, filepath)
    elif filepath.endswith('.h5'):
        model.save(filepath)
