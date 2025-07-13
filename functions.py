import pandas as pd
import numpy as np
from functools import reduce
import difflib
from datetime import datetime

def merge_country_observations_old(df, name1_list, name2_list):
    now = datetime.now()
    print(name1_list, name2_list)
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        for year in df['year'].unique():
            mean_values = df[(df['country'].isin(name2_list)) & (df['year'] == year)][column].mean()
            for country in name1_list:
                df.loc[(df['country'] == country) & (df['year'] == year), column] = df.loc[(df['country'] == country) & (df['year'] == year), column].fillna(mean_values)
    df = df.drop(df[df['country'].isin(name2_list)].index)
    then = datetime.now()
    print('Elapsed time:', (then - now).total_seconds() / 60, 'minutes')
    return df

def merge_country_observations(df, name1_list, name2_list):
    now = datetime.now()
    print(name1_list, name2_list)
    numeric_columns = df.select_dtypes(include=['number']).columns
    mean_values_by_year = df[df['country'].isin(name2_list)].groupby('year')[numeric_columns].mean()
    for country in name1_list:
        for column in numeric_columns:
            df.loc[df['country'] == country, column] = df.loc[df['country'] == country].apply(
                lambda row: row[column] 
                if pd.notna(row[column]) 
                else mean_values_by_year.loc[row['year'], column] 
                if row['year'] in mean_values_by_year.index else np.nan,  # Handle missing years
                axis=1)
    df = df[~df['country'].isin(name2_list)]
    then = datetime.now()
    print('Elapsed time:', (then - now).total_seconds(), 'seconds')
    return df

    
def standardize_dataframe(df, target, exclude=None):
    exclude = exclude or []
    for col in df.columns:
        if col != target and col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            m, s = df[col].mean(), df[col].std()
            df[col] = (df[col] - m) / s if s != 0 else 0
    return df