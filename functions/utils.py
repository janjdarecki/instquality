import pandas as pd
import numpy as np

id_cols = ["country", "year", "iso_code_1", "iso_code_2", "region"]


def standardise(train, test, add_mis=True):
    tr, te = train.copy(), test.copy()
    data_cols = [c for c in tr.columns if c not in id_cols]
    for col in data_cols:
        if add_mis:
            tr[f"{col}__mis"] = tr[col].isna().astype(int)
            te[f"{col}__mis"] = te[col].isna().astype(int)
        med = tr[col].median(skipna=True) # better for outliers
        tr[col] = tr[col].fillna(med)
        te[col] = te[col].fillna(med)
        mean = tr[col].mean(skipna=True)
        std  = tr[col].std(skipna=True)
        std = std if std and pd.notna(std) else 1.0
        tr[col] = (tr[col] - mean) / std
        te[col] = (te[col] - mean) / std
    return tr, te


def check_standardisation(df, name=""):
    data_cols = [c for c in df.columns if c not in id_cols and not c.endswith("__mis")]
    means = df[data_cols].mean().round(3)
    stds  = df[data_cols].std().round(3)
    print(f"— {name} —")
    print("Mean range:", means.min(), "→", means.max())
    print("Std  range:", stds.min(), "→", stds.max())
    print()


def assert_zero_leakage(tr, te, tr_s, te_s, name, split_year):
    assert tr["year"].max() <= split_year, f"{name}: training years exceed split"
    assert te["year"].min()  > split_year,  f"{name}: test years not strictly after split"
    assert set(tr["year"]).isdisjoint(set(te["year"])), f"{name}: overlapping years"
    for df, label in [(tr, f"{name}_train"), (te, f"{name}_test")]:
        assert not df.duplicated(["country","year"]).any(), f"Duplicates in {label}"
    assert set(zip(tr.country, tr.year)).isdisjoint(set(zip(te.country, te.year))), f"{name}: overlapping country-year pairs"
    feature_cols = [c for c in tr_s.columns if c not in id_cols and c != "year" and not c.endswith("__mis")]
    scaled_train_means = tr_s[feature_cols].mean()
    scaled_test_means  = te_s[feature_cols].mean()
    assert np.allclose(scaled_train_means.mean(), 0, atol=1e-6), f"{name}: train not centred at 0"
    assert not np.allclose(scaled_test_means.mean(), 0, atol=1e-6), f"{name}: test seems rescaled on itself"

    
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
    df['tgt_spread_lag'] = df.groupby('country')['tgt_spread'].shift(-1)
    df.insert(5, 'tgt_spread_lag', df.pop('tgt_spread_lag'))
    df = df.drop(columns=['tgt_spread', 'tgt_yield'])
    return df
