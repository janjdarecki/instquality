import pandas as pd
import numpy as np
from functools import reduce

def load():
    # Economic freedom in the world
    efw = pd.read_excel("IQD/EFW.xlsx", header=4, usecols=lambda col: col != 'Unnamed: 0').rename(columns={'Countries':'country'})
    efw = strip_cols(efw)
    efw = prefix_and_remove(efw, "efw", 4)

    # Freedom in the world
    fiw = pd.read_excel("IQD/FIW.xlsx", sheet_name='FIW06-23', usecols='A:S').rename(columns={'Country/Territory':'country',
                                                                                              'Edition':'year',
                                                                                              'C/T?':'fiw_c_t'})
    fiw = strip_cols(fiw)
    fiw = prefix_and_remove(fiw, "fiw", 4)

    # Index of economic freedom
    ief = pd.read_csv("IQD/IEF.csv", usecols=lambda col: col != 'Id').rename(columns={'Name':'country',
                                                                                      'Index Year':'year'})
    ief = strip_cols(ief)
    ief = prefix_and_remove(ief, "ief", 4)
    
    # Polity 5D
    p5d = pd.read_excel("IQD/P5D.xls").drop(columns=['p5', 'cyear', 'ccode', 'scode'])
    p5d = p5d[p5d.year>=1960].reset_index(drop=True)
    p5d = strip_cols(p5d)
    p5d = prefix_and_remove(p5d, "p5d", 2)
    
    # Political terror scale
    pts = pd.read_excel("IQD/PTS.xlsx").drop(columns=['COW_Code_A', 'COW_Code_N', 'WordBank_Code_A', 'UN_Code_N', 'Region'])
    pts = strip_cols(pts)
    pts = prefix_and_remove(pts, "pts", 3)
    
    # World governance indicators
    wgi1 = prep_wgi(sheet_name="VoiceandAccountability")
    wgi2 = prep_wgi(sheet_name="Political StabilityNoViolence")
    wgi3 = prep_wgi(sheet_name="GovernmentEffectiveness")
    wgi4 = prep_wgi(sheet_name="RegulatoryQuality")
    wgi5 = prep_wgi(sheet_name="RuleofLaw")
    wgi6 = prep_wgi(sheet_name="ControlofCorruption")
    
    # Merge
    dataframes = [wgi1, wgi2, wgi3, wgi4, wgi5, wgi6]
    wgi = reduce(lambda left, right: pd.merge(left, right, on=['country', 'code', 'Year'], how='left'), dataframes)
    del wgi1, wgi2, wgi3, wgi4, wgi5, wgi6
    
    # Final
    wgi = strip_cols(wgi)
    wgi = prefix_and_remove(wgi, "wgi", 3) 
    
    return efw, fiw, ief, p5d, pts, wgi
    

def strip_cols(df):
    # Adjust columns
    df.columns = (
        df.columns.str.strip()      
        .str.lower()
        .str.replace('.', '_')
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace(',', '_')
        .str.replace('?', '')           
        .str.replace('[^a-z0-9_]', '')
        .str.replace('/', '_'))
    return df
    
    
def prefix_and_remove(df, prefix, skipcols, nullpct=0.9):
    # Add prefix for columns
    df.columns = [col if i < skipcols else f"{prefix}_{col}"
        for i, col in enumerate(df.columns)]
    
    # Remove high null columns
    null_ratio = df.isnull().sum() / len(df)
    cols_to_drop = null_ratio[null_ratio > nullpct].index
    print("Dropping high-null cols", cols_to_drop)
    df = df.drop(columns=cols_to_drop)

    return df


def prep_wgi(sheet_name):
    # Load
    df = pd.read_excel("IQD/WGI.xlsx", sheet_name=sheet_name, header=[13, 14])
    
    # Prep
    df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    df = df.rename(columns={'Unnamed: 0_level_0 Country/Territory':'country',
                            'Unnamed: 1_level_0 Code':'code'})
    df_long = pd.melt(df, id_vars=['country', 'code'], var_name='Year_Metric', value_name='Value')
    df_long['Year'] = df_long['Year_Metric'].str.extract('(\d+)')[0]  # Extract year part
    df_long['Metric'] = df_long['Year_Metric'].str.split(' ').str[1]  # Extract metric name
    df_long = df_long.drop(columns=['Year_Metric'])
    df_long = df_long[['country', 'code', 'Year', 'Metric', 'Value']]
    
    # Pivot
    df_wide = df_long.pivot_table(
        index=['country', 'code', 'Year'],  # Columns to keep, not to spread
        columns='Metric',  # Column to spread
        values='Value',  # Fill values
        aggfunc='first'  # Function to aggregate values, 'first' because each combination is unique
    )
    df_wide.reset_index(inplace=True)
    df_wide.columns.name = None
    df_wide.rename(columns={col: f"{sheet_name}_{col}" for col in df_wide.columns if col not in ['country', 'code', 'Year']}, inplace=True)
    return df_wide