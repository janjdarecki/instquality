import pandas as pd
import numpy as np


def fill(df):
    df = save_nulls_share(df, 'init')
    ffilled = ffill_cols(df)
    ffilled = save_nulls_share(ffilled, 'ffill')
    ffilled = obj_to_num(ffilled)
    
    rfilled = fill_from(ffilled, 'region')
    rfilled = save_nulls_share(rfilled, 'rfill')
    
    wfilled = fill_from(rfilled[rfilled.year>=1970], 'world')
    wfilled = save_nulls_share(wfilled, 'wfill')
    
    wfilled = wfilled[[c for c in wfilled.columns if not c.startswith('nulls_')] 
        + [c for c in wfilled.columns if c.startswith('nulls_')]]
    return wfilled


def compute_distances(s):
    dist, last = [], None
    for v in s:
        if pd.notnull(v):
            last = 0
            dist.append(0)
        else:
            dist.append(np.nan if last is None else (last := last + 1))
    return pd.Series(dist, index=s.index)


def ffill_cols(df):
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    filled = df.copy()
    fixed = ['country', 'year', 'iso_code_1', 'iso_code_2', 'region', 'nulls_init']
    cols_to_fill = [c for c in df.columns if c not in fixed]
    for country in df['country'].unique():
        print(f"F-filling {country} values")
        mask = df['country'] == country
        sub = df.loc[mask]
        filled.loc[mask, cols_to_fill] = sub.ffill()
        for col in cols_to_fill:
            dist = compute_distances(sub[col])
            filled.loc[mask, f"{col}_f"] = dist.values
    return filled


def fill_from(df, source='region', penalty=10):
    df = df.copy()
    fixed = ['country', 'year', 'iso_code_1', 'iso_code_2', 'region'
            ] + [c for c in df.columns if c.startswith('nulls_')]
    data_cols = [c for c in df.columns if c not in fixed and not c.endswith('_f')]
    if source == 'region':
        group_cols = ['year', 'region']
        pen = penalty
    elif source == 'world':
        group_cols = ['year']
        pen = penalty * 25
    grouped = df.groupby(group_cols)
    for col in data_cols:
        df[col] = df[col].fillna(grouped[col].transform('mean'))
        fcol = f"{col}_f"
        if fcol in df.columns:
            mean_f  = grouped[fcol].transform('mean')
            count_n = grouped[fcol].transform('count')
            df[fcol] = df[fcol].fillna(mean_f + (pen / count_n))
    return df


def obj_to_num(df):
    for col in ['efw_data_3', 'efw_data_4', 'efw_5bv_cost_of_worker_dismissal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['fiw_c_t'] = df['fiw_c_t'].map({'c': 0, 't': 1}).astype(float)
    df['fiw_status'] = df['fiw_status'].map({'PF': 0.5, 'NF': 0, 'F': 1}).astype(float)
    return df


def save_nulls_share(df, version, no):
    fixed = ['country', 'year', 'iso_code_1', 'iso_code_2', 'region']
    data_cols = [c for c in df.columns if c not in fixed and not c.endswith('_f')]
    df[f'nulls_{version}'] = df[data_cols].isna().mean(axis=1)
    return df