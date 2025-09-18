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