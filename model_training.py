
"""model_training.py
- Loads hierarchical_sales_wide.csv
- Trains ARIMA models for each bottom-level series (RegionX_ProductY) with hyperparameter search (AIC-based grid search).
- Produces forecasts for a horizon and saves forecasts CSV.
Usage: python model_training.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def load_data(path='hierarchical_sales_wide.csv'):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def train_arima_grid(ts, p_vals=(0,1,2), d_vals=(0,1), q_vals=(0,1,2)):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                try:
                    model = ARIMA(ts, order=(p,d,q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p,d,q)
                        best_model = model
                except Exception:
                    continue
    return best_model, best_order

def forecast_hierarchy(df, horizon=12, bottom_prefixes=None):
    # bottom-level series: those with '_' in name
    if bottom_prefixes is None:
        bottom = [c for c in df.columns if '_' in c]
    else:
        bottom = bottom_prefixes
    forecasts = {}
    fitted = {}
    for col in bottom:
        ts = df[col]
        model, order = train_arima_grid(ts)
        if model is None:
            # fallback: simple naive forecast (last value)
            fc = np.repeat(ts.iloc[-1], horizon)
        else:
            fc = model.forecast(steps=horizon)
        forecasts[col] = fc
        # save fitted values for backtesting if needed
        try:
            fitted[col] = model.fittedvalues if model is not None else None
        except Exception:
            fitted[col] = None
    # convert forecasts to DataFrame (dates forward)
    last_date = df.index[-1]
    future_idx = pd.date_range(start=last_date, periods=horizon+1, freq=df.index.freq)[1:]
    fc_df = pd.DataFrame(forecasts, index=future_idx)
    fc_df.to_csv('forecasts_bottom.csv')
    print('Saved forecasts_bottom.csv')
    return fc_df, fitted

if __name__ == '__main__':
    df = load_data()
    fc, fitted = forecast_hierarchy(df)
