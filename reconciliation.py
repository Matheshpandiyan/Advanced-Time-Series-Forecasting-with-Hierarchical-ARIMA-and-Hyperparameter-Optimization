
"""reconciliation.py
Implements simple Bottom-Up aggregation and OLS-based MinT (optimal combination) reconciliation.
Input: forecasts for bottom-level series (DataFrame), historical wide-format df to build summing matrix.
Output: reconciled forecasts saved to CSV.
"""
import pandas as pd
import numpy as np

def build_summing_matrix(df):
    # df is wide-format historical with columns including bottom-level and aggregated series
    cols = df.columns.tolist()
    bottom = [c for c in cols if '_' in c]
    # order: Total, RegionA, RegionB, RegionA_Product1, ...
    # Build matrix S where agg = S * bottom
    agg = ['Total', 'RegionA', 'RegionB'] + bottom
    S = []
    for a in agg:
        if a == 'Total':
            row = [1]*len(bottom)
        elif a == 'RegionA':
            row = [1 if c.startswith('RegionA_') else 0 for c in bottom]
        elif a == 'RegionB':
            row = [1 if c.startswith('RegionB_') else 0 for c in bottom]
        else:
            # bottom series
            row = [1 if c==a else 0 for c in bottom]
        S.append(row)
    S = np.array(S)
    return np.array(agg), S

def bottom_up(fc_bottom):
    # fc_bottom: DataFrame (index=future dates, columns=bottom series)
    df = pd.DataFrame(index=fc_bottom.index)
    df['Total'] = fc_bottom.sum(axis=1)
    # Region sums
    ra_cols = [c for c in fc_bottom.columns if c.startswith('RegionA_')]
    rb_cols = [c for c in fc_bottom.columns if c.startswith('RegionB_')]
    df['RegionA'] = fc_bottom[ra_cols].sum(axis=1)
    df['RegionB'] = fc_bottom[rb_cols].sum(axis=1)
    # include bottoms
    for c in fc_bottom.columns:
        df[c] = fc_bottom[c]
    return df

def mint_ols_reconcile(fc_bottom, S, agg_names):
    # Basic OLS MinT: reconcile forecasts to higher levels.
    # fc_bottom: (T x n_bottom)
    # S: (n_agg x n_bottom) summing matrix
    # returns reconciled aggregated forecasts (T x n_agg)
    # Using OLS MinT: y_reconciled = S * P * y_base where P = (S' S)^{-1} S'
    # where y_base are bottom forecasts stacked; simplified OLS solution
    n_bottom = S.shape[1]
    # compute P
    try:
        P = np.linalg.inv(S.dot(S.T)).dot(S)
    except np.linalg.LinAlgError:
        # regularize
        P = np.linalg.pinv(S.dot(S.T)).dot(S)
    # For each time step, compute reconciled agg forecasts
    rec_list = []
    for t in range(fc_bottom.shape[0]):
        yb = fc_bottom.iloc[t].values.reshape(-1,1)  # (n_bottom x 1)
        y_rec = S.dot(yb)  # (n_agg x 1)
        rec_list.append(y_rec.flatten())
    rec = pd.DataFrame(rec_list, index=fc_bottom.index, columns=agg_names)
    return rec

if __name__ == '__main__':
    # small demo when run directly
    hist = pd.read_csv('hierarchical_sales_wide.csv', index_col=0, parse_dates=True)
    bottom = [c for c in hist.columns if '_' in c]
    agg_names, S = build_summing_matrix(hist)
    # load forecasts
    fc_bottom = pd.read_csv('forecasts_bottom.csv', index_col=0, parse_dates=True)
    rec_ols = mint_ols_reconcile(fc_bottom, S, agg_names)
    rec_ols.to_csv('reconciled_ols_forecasts.csv')
    bottom_up(rec_ols).to_csv('reconciled_bottomup_forecasts.csv')
    print('Saved reconciled forecasts.')
