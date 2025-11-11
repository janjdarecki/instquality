import pandas as pd
import numpy as np
import re

id_cols = ["country", "year", "iso_code_1", "iso_code_2", "region"]


def handle_p5d_vals(df):
    # save transition indicators
    df["p5d_trans_indicator"] = np.select(
        [df["p5d_polity"] == -77,  # interregnum, mark as 1
        df["p5d_polity"] == -88,  # transition, mark as 2
        df["p5d_polity"] == -99   # foreign interruption, mark as 3
        ], [1, 2, 3],  # codes for each case
        default=0   # normal years
    )
    
    # force nans on rest
    p5d_cols = [c for c in df.columns if c.startswith(f"p5d_")]
    df[p5d_cols] = df[p5d_cols].replace([-99, -88, -77], np.nan) # replace the values
    return df


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


def check_split_year(df, yname, share=0.75):
    dff = df.dropna(subset=[yname]).sort_values("year")
    dff["cum_share"] = (dff.reset_index().index + 1) / len(dff)
    split_year = dff.loc[dff["cum_share"] >= share, "year"].iloc[0]
    return split_year
    

def engineer_lag_vars(df, macro_vars, iq_vars, id_cols=["country", "year", "iso_code_1", "iso_code_2", "region"]):
    df = df.sort_values(id_cols).copy()
    n_before = df.shape[1]

    # Deltas
    for var in macro_vars + iq_vars:
        df[f"{var}_delta"] = df.groupby("country")[var].diff(1)
        df[f"{var}_delta3"] = df.groupby("country")[var].diff(3)

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


def remove_highly_correlated(df, threshold):
    id_cols = ["country", "year", "iso_code_1", "iso_code_2", "region"]
    exclude = id_cols + [c for c in df.columns if c.startswith("tgt_")]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    df_features = df[feature_cols]
    corr_matrix = df_features.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_triangle)
    to_drop = set()
    
    for column in upper_corr.columns:
        high_corr = upper_corr[column][upper_corr[column] > threshold]
        for corr_col in high_corr.index:
            if corr_col not in to_drop and column not in to_drop:
                col_count = df_features[column].notna().sum()
                corr_col_count = df_features[corr_col].notna().sum()
                if col_count > corr_col_count:
                    to_drop.add(corr_col)
                else:
                    to_drop.add(column)
    
    cols_to_drop = sorted(list(to_drop))
    return cols_to_drop

