import pandas as pd
import numpy as np
import os
import zipfile
from functools import reduce


def load(folder="pickles"):
    cfg=[
        ("efw", "EFW.xlsx", "excel", {'header':4,'usecols':range(1,85)}, ("Countries", "Year", "ISO Code 2", "ISO Code 3", "World Bank Region"), None),
        ("fiw", "FIW.xlsx", "excel", {'sheet_name':'FIW06-23','usecols':'A:S'}, ("Country/Territory", "Edition", None, None, "Region"), None),
        ("ief", "IEF.csv", "csv", {'usecols':lambda c:c not in ['Id', 'Short Name']}, ("Name", "Index Year", "ISO Code"), None),
        ("p5d","P5D.xls","excel", {'usecols':lambda c:c not in [
             'scode', 'p5', 'cyear', 'ccode', 'bmonth', 'byear', 'bday',
             'prior', 'emonth', 'eday', 'eyear', 'eprec', 'interim',
             'bprec', 'post', 'change', 'd5', 'sf']}, ("country", "year", None, None, None), lambda df:df[df.year>=1960]),
        ("pts", "PTS.xlsx", "excel", {'usecols':lambda c:c not in ['Country_OLD', 'COW_Code_A', 'COW_Code_N', 'UN_Code_N']}, ("Country", "Year", None, "WordBank_Code_A", "Region"), None),
        ("wgi", "WGI.xlsx", "wgi", {}, ("country", "Year", None, "code", None), None),
        ("wb", "WB_DATA.zip", "wb", {}, ("country_name", "year", None, "country_code", None), None),
    ]
    
    res={}
    for k,f,typ,opts,ren,pf in cfg:
        path=os.path.join(folder,f)
        print(f"Loading {path}")
        if typ=='csv': df=pd.read_csv(path,**opts)
        elif typ=='excel': df=pd.read_excel(path,**opts)
        elif typ=='wgi': df=load_wgi(path)
        else: df=load_wb(path)
        if pf: df=pf(df)
        # Functions
        df = rename_cols(df, *ren)
        df = strip_cols(df)
        df = prefix_cols(df, k)
        df = ffill_cols(df)
        res[k] = df
    
    return res


def strip_cols(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[.\s\-/,]", "_", regex=True)       # replace separators with _
        .str.replace(r"\?", "", regex=True)               # remove ?
        .str.replace(r"[^a-z0-9_]", "", regex=True)       # remove other non-alphanum
        .str.replace(r"_+", "_", regex=True)              # collapse multiple underscores
        .str.strip("_")                                   # remove leading/trailing _
    )
    return df


def rename_cols(df, country, year, iso_code_1=None, iso_code_2=None, region=None):
    df = df.rename(columns={
        country: 'country',
        year: 'year',
        **({iso_code_1: 'iso_code_1'} if iso_code_1 else {}),
        **({iso_code_2: 'iso_code_2'} if iso_code_2 else {}),
        **({region: 'region'} if region else {})})
    
    if 'iso_code_1' not in df:
        df['iso_code_1'] = np.nan
    if 'iso_code_2' not in df:
        df['iso_code_2'] = np.nan
    if 'region' not in df:
        df['region'] = np.nan
        
    df = df[['country', 'year', 'iso_code_1', 'iso_code_2', 'region'] + [c for c in df.columns if c not in {'country', 'year', 'iso_code_1', 'iso_code_2', 'region'}]]
    df = df.sort_values(by=['country', 'year']).reset_index(drop=True)
    return df


def prefix_cols(df, prefix):
    df.columns = [col if i < 5 else f"{prefix}_{col}"
    for i, col in enumerate(df.columns)]
    return df


def ffill_cols(df):
    df = df.sort_values(by=['country', 'year']).reset_index(drop=True)
    df[df.columns] = (df.groupby(['country'])[df.columns].transform(lambda group: group.ffill()))
    return df


def load_wgi(filename):
    wgi1 = prep_wgi(filename, "VoiceandAccountability")
    wgi2 = prep_wgi(filename, "Political StabilityNoViolence")
    wgi3 = prep_wgi(filename, "GovernmentEffectiveness")
    wgi4 = prep_wgi(filename, "RegulatoryQuality")
    wgi5 = prep_wgi(filename, "RuleofLaw")
    wgi6 = prep_wgi(filename, "ControlofCorruption")

    dataframes = [wgi1, wgi2, wgi3, wgi4, wgi5, wgi6]
    wgi = reduce(lambda left, right: pd.merge(left, right, on=['country', 'code', 'Year'], how='left'), dataframes)
    del wgi1, wgi2, wgi3, wgi4, wgi5, wgi6
    
    return wgi


def prep_wgi(filename, sheet_name):
    df = pd.read_excel(filename, sheet_name=sheet_name, header=[13, 14])
    df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    df = df.rename(columns={'Unnamed: 0_level_0 Country/Territory':'country',
                            'Unnamed: 1_level_0 Code':'code'})
    df_long = pd.melt(df, id_vars=['country', 'code'], var_name='Year_Metric', value_name='Value')
    df_long['Year'] = df_long['Year_Metric'].str.extract('(\d+)')[0]  # Extract year part
    df_long['Metric'] = df_long['Year_Metric'].str.split(' ').str[1]  # Extract metric name
    df_long = df_long.drop(columns=['Year_Metric'])
    df_long = df_long[['country', 'code', 'Year', 'Metric', 'Value']]
    
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


def load_wb(zipname):
    with zipfile.ZipFile("pickles/WB_DATA.zip") as z:   # open the .zip
        with z.open("WB_DATA_d950d0cd269a601150c0afd03b234ee2.csv") as f:      # open the CSV file inside
            wb = pd.read_csv(f)            # read it straight into a DataFrame
            wb = wb.pivot_table(index=['country_code', 'country_name', 'year'], columns='series_id', values='value').reset_index() # and pivot straight away

    return wb