
"""evaluation.py
Computes RMSE and MASE for forecasts against actuals.
MASE implementation uses naive seasonal or non-seasonal naive based on frequency.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mase(y_true, y_pred, y_train, seasonality=12):
    # Mean Absolute Scaled Error
    n = len(y_train)
    if n <= seasonality:
        # fallback to non-seasonal naive
        d = np.abs(np.diff(y_train)).mean()
    else:
        d = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    errors = np.abs(y_true - y_pred).mean()
    return errors / d if d != 0 else np.inf

def evaluate_forecasts(actual_df, fc_df, train_df):
    # actual_df and fc_df have same structure: columns of series; index aligned (future dates)
    results = []
    for col in fc_df.columns:
        y_pred = fc_df[col].values
        y_true = actual_df[col].values if col in actual_df.columns else np.full_like(y_pred, np.nan)
        train_series = train_df[col].values if col in train_df.columns else np.array([])
        rm = rmse(y_true, y_pred)
        ma = mase(y_true, y_pred, train_series)
        results.append({'series': col, 'RMSE': float(rm), 'MASE': float(ma)})
    return pd.DataFrame(results)

if __name__ == '__main__':
    # demo usage
    hist = pd.read_csv('hierarchical_sales_wide.csv', index_col=0, parse_dates=True)
    fc = pd.read_csv('reconciled_ols_forecasts.csv', index_col=0, parse_dates=True)
    # If actuals for forecast horizon not available in demo, this will illustrate function usage.
    print('Evaluation module ready.')
