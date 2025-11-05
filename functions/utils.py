import pandas as pd
import numpy as np

id_cols = ["country", "year", "iso_code_1", "iso_code_2", "region"]


def prep_target(df):
    # create spread var
    us_yields = (
        df.loc[df['country'] == 'United States', ['year', 'tgt_yield']]
        .set_index('year')['tgt_yield']
    )
    df['tgt_spread'] = df['year'].map(us_yields)
    df['tgt_spread'] = df['tgt_yield'] - df['tgt_spread']
    df.insert(5, 'tgt_spread', df.pop('tgt_spread'))
    df = df.sort_values(['country', 'year'])
    df = df[df.country != 'United States']  # remove US as an observation
    
    # create lags t+1 to t+10
    for i in range(1, 11):
        df[f'tgt_spread_t{i}'] = df.groupby('country')['tgt_spread'].shift(-i)
    
    # reorder columns to place lagged vars after tgt_spread
    cols = df.columns.tolist()
    tgt_spread_idx = cols.index('tgt_spread')
    lag_cols = [f'tgt_spread_t{i}' for i in range(1, 11)]
    other_cols = [c for c in cols if c not in lag_cols + ['tgt_spread']]
    new_order = (
        other_cols[:tgt_spread_idx] + ['tgt_spread'] + lag_cols + other_cols[tgt_spread_idx:]
    )
    return df[new_order]


def engineer_lag_vars(df, macro_vars, iq_vars, id_cols=["country", "year", "iso_code_1", "iso_code_2", "region"]):
    df = df.sort_values(id_cols).copy()
    n_before = df.shape[1]

    # Lagged levels 
    for lag in [1, 3, 5]:
        for var in macro_vars + iq_vars:
            df[f"{var}_t-{lag}"] = df.groupby("country")[var].shift(lag)
    
    # Rolling averages 
    for win in [3, 5, 10]:
        for var in macro_vars + iq_vars:
            df[f"{var}_ma{win}"] = (
                df.groupby("country")[var]
                  .rolling(win, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
    
    n_after = df.shape[1]
    print(f"Added {n_after - n_before} engineered columns")
    return df

