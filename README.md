
# Advanced Time Series Forecasting with Hierarchical ARIMA and Hyperparameter Optimization

## Contents
- `data_generation.py` - generates a synthetic hierarchical dataset (Total -> Region -> Product)
- `model_training.py` - trains ARIMA models on bottom-level series using AIC-based grid-search and saves bottom forecasts
- `reconciliation.py` - provides Bottom-Up and a simple OLS-based MinT reconciliation implementation
- `evaluation.py` - functions to compute RMSE and MASE and produce comparison tables
- `hierarchical_arima.ipynb` - notebook walkthrough (basic)
- `Advanced_Time_Series_Forecasting_Project.zip` - packaged deliverable (this zip)

## How to run
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy pandas statsmodels scikit-learn matplotlib
```

2. Generate data:
```bash
python data_generation.py
```

3. Train models and forecast:
```bash
python model_training.py
```

4. Reconcile forecasts:
```bash
python reconciliation.py
```

5. Evaluate (if actuals available for horizon):
```bash
python evaluation.py
```

## Notes
- The ARIMA hyperparameter search uses grid-search over reasonable p/d/q values and selects models by AIC.
- If `scikit-optimize` or other hyperparameter tools are available, the model_training.py can be extended to use Bayesian optimization.
- MinT implemented here is a simplified OLS MinT for demonstration; for robust variance estimation, consider using sample covariance of residuals.
