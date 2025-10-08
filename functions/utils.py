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
