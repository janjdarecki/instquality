import pandas as pd
import numpy as np
import country_converter as coco
from functools import reduce


def merge(dfs, conc):
    merged = reduce(lambda left, right: pd.merge(left,
                                                 right.drop(columns='country'),
                                                 on=['year', 'iso_code_1', 'iso_code_2', 'region'],
                                                 how='outer'), dfs)
    
    merged['country'] = merged.groupby('iso_code_1')['country'].transform(lambda col: col.ffill().bfill())
    conc = conc[['country', 'iso_code_1']].drop_duplicates(subset='iso_code_1').rename(columns={'country':'country_fill'})
    merged = merged.merge(conc, on='iso_code_1', how='left')
    merged['country'] = merged['country'].fillna(merged['country_fill'])
    merged = merged.drop(columns='country_fill')
    merged = merged.sort_values(by=['country', 'year']).reset_index(drop=True)
    
    merged.country = merged.country.replace("Yemen, Rep.", "Yemen")
    merged.country = merged.country.replace("Bahamas, The", "Bahamas")
    merged.country = merged.country.replace("Egypt, Arab Rep.", "Egypt")
    merged.country = merged.country.replace("Gambia, The", "Gambia")
    merged.country = merged.country.replace("German Democratic Republic", "Germany East")
    merged.country = merged.country.replace("Iran, Islamic Rep.", "Iran")
    merged.country = merged.country.replace("Kyrgyz Republic", "Kyrgyzstan")
    merged.country = merged.country.replace("Lao PDR", "Laos")
    merged.country = merged.country.replace("Palestine, State of", "Palestine")
    merged.country = merged.country.replace("Slovak Republic", "Slovakia")
    merged.country = merged.country.replace("Russian Federation", "Russia")
    merged.country = merged.country.replace("Syrian Arab Republic", "Syria")
    merged.country = merged.country.replace("Venezuela, RB", "Venezuela")
    merged.country = merged.country.replace("Korea North", "North Korea")
    merged.country = merged.country.replace("Korea, Rep.", "South Korea")
    
    return merged


def pop_col(df, col, no):
    cols = list(df.columns)
    cols.insert(no, cols.pop(cols.index(col)))
    df = df[cols]
    return df


def apply_concordance(df, conc, name):
    df['country'] = df['country'].str.strip()
    df.year = df.year.astype("int64")
    df = df.drop(['iso_code_1', 'iso_code_2', 'region'], axis=1).merge(conc, on='country', how='left')
    
    rm = df.loc[df['iso_code_1'].isna() | df['iso_code_2'].isna(), 'country'].unique()
    print(f"Removed from {name}:", rm)
    
    df = df.dropna(subset=['iso_code_1', 'iso_code_2'])
    cols = ['country', 'year', 'iso_code_1', 'iso_code_2', 'region'] + [c for c in df.columns if c not in {'country', 'year', 'iso_code_1', 'iso_code_2', 'region'}]
    return df[cols]


def create_concordance(dfs):
    combined = pd.concat([df[['country', 'iso_code_1', 'iso_code_2']
                            ] for df in dfs], ignore_index=True)
    
    collapsed = (combined.groupby('country', as_index=False).agg({
            'iso_code_1': lambda s: s.dropna().unique()[0] if len(s.dropna()) else None,
            'iso_code_2': lambda s: s.dropna().unique()[0] if len(s.dropna()) else None}))

    collapsed = drop_regions(collapsed)
    collapsed = find_isos(collapsed)

    concordance = (collapsed.groupby(['iso_code_1', 'iso_code_2'], as_index=False)
                   .agg(country=('country', lambda names: '; '.join(names))))
    
    concordance = (concordance.assign(country = concordance['country'].str.split(';')).explode('country'))
    concordance['country'] = concordance['country'].str.strip()
    
    concordance['region'] = to_region(concordance['iso_code_1'], target='UNregion', src='ISO2')
    missing_regions = {'CS': 'Southern Europe',
                       'DD': 'Western Europe',
                       'SU': 'Eastern Europe',
                       'YU': 'Southern Europe'}
    concordance['region'] = concordance['region'].fillna(concordance['iso_code_1'].map(missing_regions))
                   
    return concordance


def find_isos(df):
    df['iso_code_1'] = df['iso_code_1'].fillna(pd.Series(to_iso(df['country'], 'ISO2'), index=df.index))
    df['iso_code_2'] = df['iso_code_2'].fillna(pd.Series(to_iso(df['country'], 'ISO3'), index=df.index))

    df.loc[df['iso_code_2'] == 'LTU', 'iso_code_1'] = 'LT'
    df.iso_code_2 = df.iso_code_2.replace("ZAR", "COD")
    df.iso_code_2 = df.iso_code_2.replace("WBG", "PSE")
    
    custom_isos = {
    'German Democratic Republic':                {'iso_code_2': 'DDR', 'iso_code_1': 'DD'},
    'Germany East':                              {'iso_code_2': 'DDR', 'iso_code_1': 'DD'},
    'German Federal Republic':                   {'iso_code_2': 'DEU', 'iso_code_1': 'DE'},
    'Germany West':                              {'iso_code_2': 'DEU', 'iso_code_1': 'DE'},
    'Serbia and Montenegro':                     {'iso_code_2': 'SCG', 'iso_code_1': 'CS'},
    'Soviet Union':                              {'iso_code_2': 'SUN', 'iso_code_1': 'SU'},
    'USSR':                                      {'iso_code_2': 'SUN', 'iso_code_1': 'SU'},
    'UAE':                                       {'iso_code_2': 'ARE', 'iso_code_1': 'AE'},
    'Yugoslavia':                                {'iso_code_2': 'YUG', 'iso_code_1': 'YU'},
    'Yugoslavia, Federal Republic of':           {'iso_code_2': 'YUG', 'iso_code_1': 'YU'},
    'Yugoslavia, Socialist Federal Republic of': {'iso_code_2': 'YUG', 'iso_code_1': 'YU'}}

    for country_name, codes in custom_isos.items():
        mask = df['country'] == country_name
        df.loc[mask, ['iso_code_1', 'iso_code_2']] = (
            codes['iso_code_1'],
            codes['iso_code_2'])
            
    return df


def to_iso(series, target):
    return coco.convert(series, to=target, not_found=np.nan)


def to_region(series, target='UNregion', src=None):
    out = coco.convert(series.tolist(), to=target, src=src, not_found=np.nan)
    return pd.Series(out, index=series.index, name=target)


def drop_regions(df):
    df = df[~df.country.isin(['(SDG) Central/Southern Asia', '(SDG) Eastern/South Eastern Asia',
       '(SDG) Europe', '(SDG) Latin America & the Caribbean',
       '(SDG) Northern America', '(SDG) Oceania',
       '(SDG) Sub-Saharan Africa', '(SDG) Western Asia/Northern Africa',
       '(UN) Africa', '(UN) Asia', '(UN) Europe',
       '(UN) Latin America and the Caribbean', '(UN) North America',
       '(UN) Oceania', '(WHO) Africa Region', '(WHO) America Region',
       '(WHO) Eastern Mediterranean Region', '(WHO) European Region',
       '(WHO) South-East Asia Region', '(WHO) Western Pacific Region',
       'Abkhazia', 'Africa Eastern and Southern',
       'Africa Western and Central',
       'American Samoa', 'Anguilla',
       'Antigua and Barbuda', 'Arab League states', 'Arab World',
       'Aruba', 'British Virgin Islands', 
       'Caribbean small states', 'Cayman Islands', 'Central Europe and the Baltics',
       'Channel Islands', 'Chechnya',
       'Cook Islands', 'Crimea',
       'Early-demographic dividend', 'East Asia & Pacific',
       'East Asia & Pacific (IDA & IBRD countries)',
       'East Asia & Pacific (IDA & IBRD)',
       'East Asia & Pacific (all income levels)',
       'East Asia & Pacific (excluding high income)', 'Eastern Donbas',
       'Euro area', 'Europe & Central Asia',
       'Europe & Central Asia (IDA & IBRD countries)',
       'Europe & Central Asia (IDA & IBRD)',
       'Europe & Central Asia (all income levels)',
       'Europe & Central Asia (excluding high income)', 'European Union',
       'Fragile and conflict affected situations', 
       'Gaza (Hamas)', 'Gaza Strip', 'Guam', 'Greenland',
       'Heavily indebted poor countries (HIPC)', 'High income',
       'High income: OECD', 'High income: nonOECD', 
       'IBRD only', 'IDA & IBRD total', 'IDA blend',
       'IDA only', 'IDA total', 'Indian Kashmir',
       'Isle of Man','Israel in Occupied Territories',
       'Israel in pre-1967 borders', 'Israeli Occupied Territories',
       'Jersey, Channel Islands', 'Kiribati',
       'Late-demographic dividend', 'Latin America & Caribbean',
       'Latin America & Caribbean (IDA & IBRD)',
       'Latin America & Caribbean (all income levels)',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification',
       'Low & middle income', 'Low income', 'Lower middle income',
       'Marshall Islands', 'Micronesia', 'Micronesia, Fed. Sts.',
       'Micronesia, Federated States of',
       'Middle East & North Africa',
       'Middle East & North Africa (IDA & IBRD countries)',
       'Middle East & North Africa (IDA & IBRD)',
       'Middle East & North Africa (all income levels)',
       'Middle East & North Africa (excluding high income)',
       'Middle income', 'Nagorno-Karabakh', 'Netherlands Antilles',
       'Netherlands Antilles (former)', 'New Caledonia', 'Niue',
       'North America', 'Northern Cyprus', 'Northern Mariana Islands', 
       'OECD members', 'Other small states',
       'Pacific island small states', 'Palau',
       'Pakistani Kashmir', 'Post-demographic dividend',
       'Pre-demographic dividend', 'Russia-Occupied Areas (Ukraine)',
       'Réunion', 'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa',
       'Sint Maarten (Dutch part)',
       'Small states', 'Solomon Islands', 'Somaliland', 'South Asia',
       'South Asia (IDA & IBRD)', 'South Ossetia',
       'South Vietnam', 'St. Kitts and Nevis', 'St. Lucia',
       'St. Martin (French part)', 'St. Vincent and the Grenadines',
       'Sub-Saharan Africa', 'Sub-Saharan Africa (IDA & IBRD countries)',
       'Sub-Saharan Africa (IDA & IBRD)',
       'Sub-Saharan Africa (all income levels)',
       'Sub-Saharan Africa (excluding high income)', 'Tibet',
       'Tonga', 'Transnistria', 'Turks and Caicos Islands',
       'Tuvalu', 
       'Upper middle income', 
       'Vanuatu', 'Vietnam North', 'Virgin Islands (U.S.)', 
       'West Bank and Gaza', 'World', 'Yemen North',
       'Yemen South'])]
    return df