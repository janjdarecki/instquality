import pandas as pd
import numpy as np


def standardise(df):
    df = df.copy()
    fixed = ['country', 'year', 'iso_code_1', 'iso_code_2', 'region']
    data_cols = [c for c in df.columns if c not in fixed and not c.endswith('_f')]
    fcols     = [c for c in df.columns if c.endswith('_f')]
    for col in data_cols:
        mn, mx = df[col].min(skipna=True), df[col].max(skipna=True)
        df[col] = ((df[col] - mn) / (mx - mn)) * 2 - 1
        df[col].fillna(0, inplace=True)
    for col in fcols:
        df[col].fillna(100, inplace=True)
        mn, mx = df[col].min(skipna=True), df[col].max(skipna=True)
        df[col] = ((df[col] - mn) / (mx - mn)) * 2 - 1
    df = df.drop(columns=['wb_dt_nfl_unep_cd', 'wb_dt_nfl_unid_cd',
                         'wb_dt_nfl_unep_cd_f', 'wb_dt_nfl_unid_cd_f'], errors='ignore')
    return df
    

def pop_col(df, col, no):
    cols = list(df.columns)
    cols.insert(no, cols.pop(cols.index(col)))
    df = df[cols]
    return df


def revert_to_raw(df):
    df = df.copy()
    # force nulls whenever _f != 0
    for col in [c for c in df.columns if c.endswith("_f")]:
        base = col[:-2]  # strip "_f"
        if base in df.columns:   # only set nulls if base column actually exists
            df.loc[df[col] != 0, base] = pd.NA
    
    # drop all *_f and nulls_* columns
    drop_cols = [c for c in df.columns if c.endswith("_f") or c.startswith("nulls_")]
    return df.drop(columns=drop_cols, errors="ignore")


def prep_target(df):
    us_yields = df.loc[df['country'] == 'United States', ['year', 'tgt_yield']].set_index('year')['tgt_yield']
    df['tgt_spread'] = df['year'].map(us_yields)
    df['tgt_spread'] = df['tgt_yield'] - df['tgt_spread']

    df = df.sort_values(['country', 'year'])
    df['tgt_yield_lag'] = df.groupby('country')['tgt_yield'].shift(-1)
    df['tgt_spread_lag'] = df.groupby('country')['tgt_spread'].shift(-1)

    df.insert(5, 'tgt_yield', df.pop('tgt_yield'))
    df.insert(6, 'tgt_spread', df.pop('tgt_spread'))
    df.insert(7, 'tgt_yield_lag', df.pop('tgt_yield_lag'))
    df.insert(8, 'tgt_spread_lag', df.pop('tgt_spread_lag'))
    return df
