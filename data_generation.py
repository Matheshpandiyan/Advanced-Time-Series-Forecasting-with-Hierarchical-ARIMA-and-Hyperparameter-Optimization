
"""data_generation.py
Generates a synthetic hierarchical time series dataset suitable for hierarchical forecasting experiments.
Structure:
- Total -> Regions -> Products
Outputs a CSV file 'hierarchical_sales.csv' with columns: date, series_id, value
and a wide-format CSV 'hierarchical_sales_wide.csv' where columns are series names.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_hierarchy(start='2015-01-01', periods=120, freq='M', seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    # hierarchy: Total -> RegionA, RegionB -> Product1/Product2 each
    series = {}
    # base seasonal pattern and trend
    t = np.arange(periods)
    base_trend = 50 + 0.5 * t
    season = 10 * np.sin(2 * np.pi * (t % 12) / 12)
    # Region multipliers
    regions = {'RegionA': 0.6, 'RegionB': 0.4}
    products = {'Product1': 0.7, 'Product2': 0.3}
    for r, r_mult in regions.items():
        for p, p_mult in products.items():
            name = f"{r}_{p}"
            noise = np.random.normal(scale=3 + 0.1 * t, size=periods)
            series[name] = (base_trend * r_mult * p_mult) + season * (1 + 0.1 * r_mult) + noise
            # ensure positive
            series[name] = np.maximum(series[name], 0.1)
    # aggregate levels
    df = pd.DataFrame(series, index=dates)
    df['RegionA'] = df[[c for c in df.columns if c.startswith('RegionA_')]].sum(axis=1)
    df['RegionB'] = df[[c for c in df.columns if c.startswith('RegionB_')]].sum(axis=1)
    df['Total'] = df[['RegionA', 'RegionB']].sum(axis=1)
    # long-format
    long = df.reset_index().melt(id_vars='index', var_name='series_id', value_name='value')
    long = long.rename(columns={'index': 'date'})
    out_dir = Path('.')
    long.to_csv(out_dir / 'hierarchical_sales.csv', index=False)
    df.to_csv(out_dir / 'hierarchical_sales_wide.csv')
    print('Generated hierarchical_sales.csv and hierarchical_sales_wide.csv in', out_dir.resolve())

if __name__ == '__main__':
    generate_hierarchy()
