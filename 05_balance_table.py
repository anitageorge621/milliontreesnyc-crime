#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_balance_table.py
MillionTreesNYC — Balance Table
=================================

"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%%

# Census API Key
with open('apikey.txt', 'r') as f:
    CENSUS_API_KEY = f.read().strip()

# ── ACS Variable Definitions ───────────────────────────────────────────────────
# Using 2006-2010 ACS 5-year estimates (vintage year = 2010)
# B15002 used for education (B15003 not available until 2011 vintage)
# B23001 used for unemployment (B23025 not available until 2011 vintage)

# Shared variables
ACS_SHARED = [
    "B19013_001E",                          # median household income
    "B17001_002E", "B17001_001E",           # below poverty / total
    "B02001_002E", "B02001_003E",           # white alone, black alone
    "B03002_012E",                          # hispanic or latino
    "B01001_001E",                          # total population
    "B01003_001E",                          # total pop for density
    "B15002_001E",                          # education universe 25+
    "B15002_015E", "B15002_016E",           # male: BA, MA
    "B15002_017E", "B15002_018E",           # male: professional, doctorate
    "B15002_032E", "B15002_033E",           # female: BA, MA
    "B15002_034E", "B15002_035E",           # female: professional, doctorate
]

# Unemployment variables from B23001 (call 2 — 40 cols)
UNEMP_COLS = (
    [f"B23001_{c:03d}E" for c in [8,15,22,29,36,43,50,57,64,71]] +
    [f"B23001_{c:03d}E" for c in [94,101,108,115,122,129,136,143,150,157]]
)
LF_COLS = (
    [f"B23001_{c:03d}E" for c in [6,13,20,27,34,41,48,55,62,69]] +
    [f"B23001_{c:03d}E" for c in [92,99,106,113,120,127,134,141,148,155]]
)
ACS_UNEMP = UNEMP_COLS + LF_COLS

NYC_COUNTIES = "005,047,061,081,085"

#%%

# API Helpers 
def single_fetch(year, var_list, api_key):
    """Fetch up to ~49 ACS variables for all NYC tracts in one API call."""
    var_str = ",".join(var_list)
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get={var_str}"
        f"&for=tract:*"
        f"&in=state:36+county:{NYC_COUNTIES}"
        f"&key={api_key}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Census API error {r.status_code}: {r.text[:300]}")

    data = r.json()
    df   = pd.DataFrame(data[1:], columns=data[0])

    # Build 11-character GEOID from state + county + tract
    df['GEOID'] = (df['state'] + df['county'] + df['tract']).str.zfill(11)

    # Convert all ACS columns to numeric and replace Census missing codes
    non_id = [c for c in df.columns
              if c not in ('state', 'county', 'tract', 'GEOID')]
    df[non_id] = df[non_id].apply(pd.to_numeric, errors='coerce')
    df[non_id] = df[non_id].replace({
        -666666666: np.nan, -999999999: np.nan,
        -888888888: np.nan, -222222222: np.nan
    })
    return df[['GEOID'] + non_id]


def fetch_acs(year, api_key):
    """Fetch all needed ACS variables, splitting into two calls to stay under limit."""
    df_main  = single_fetch(year, ACS_SHARED, api_key)
    df_unemp = single_fetch(year, ACS_UNEMP,  api_key)
    return df_main.merge(df_unemp, on='GEOID', how='left')


def build_acs_features(df):
    """Derive analysis variables from raw ACS columns."""
    out = df[['GEOID']].copy()

    # Median household income
    out['median_hh_income'] = df['B19013_001E']

    # Poverty rate
    out['poverty_rate'] = (
        df['B17001_002E'] / df['B17001_001E'].replace(0, np.nan)
    )

    # Total population
    out['total_population'] = df['B01003_001E']

    # Race/ethnicity shares
    out['share_white']    = df['B02001_002E'] / df['B01001_001E'].replace(0, np.nan)
    out['share_black']    = df['B02001_003E'] / df['B01001_001E'].replace(0, np.nan)
    out['share_hispanic'] = df['B03002_012E'] / df['B01001_001E'].replace(0, np.nan)

    # Share with BA or higher (population 25+)
    ba_plus = (
        df['B15002_015E'] + df['B15002_016E'] +
        df['B15002_017E'] + df['B15002_018E'] +
        df['B15002_032E'] + df['B15002_033E'] +
        df['B15002_034E'] + df['B15002_035E']
    )
    out['share_ba_plus'] = ba_plus / df['B15002_001E'].replace(0, np.nan)

    # Unemployment rate (B23001 — available for 2010 vintage)
    unemp_total = df[[c for c in UNEMP_COLS if c in df.columns]].sum(axis=1)
    lf_total    = df[[c for c in LF_COLS    if c in df.columns]].sum(axis=1)
    out['unemp_rate'] = unemp_total / lf_total.replace(0, np.nan)

    return out

#%%

# 1. Fetching ACS 2006-2010 5-year estimates

raw_acs = fetch_acs(2010, CENSUS_API_KEY)
acs     = build_acs_features(raw_acs)

print(f"   {len(acs)} tracts fetched")

#%%

# 2. Merging with treatment status

panel = pd.read_csv("regression_panel_annual.csv")

tract_treatment = (
    panel.groupby('GEOID')
    .agg(ever_treated = ('ever_treated', 'first'))
    .reset_index()
)
tract_treatment['GEOID'] = tract_treatment['GEOID'].astype(str).str.zfill(11)
acs['GEOID']             = acs['GEOID'].astype(str).str.zfill(11)

merged = acs.merge(tract_treatment, on='GEOID', how='inner')
print(f"   {len(merged)} tracts matched")
print(f"   Treated:       {merged['ever_treated'].sum()}")
print(f"   Never-treated: {(merged['ever_treated']==0).sum()}")

#%%

# 3. Build Balance Table 


BALANCE_VARS = {
    'median_hh_income': 'Median Household Income ($)',
    'poverty_rate':     'Poverty Rate',
    'unemp_rate':       'Unemployment Rate',
    'total_population': 'Total Population',
    'share_white':      'Share White',
    'share_black':      'Share Black',
    'share_hispanic':   'Share Hispanic',
    'share_ba_plus':    'Share BA or Higher',
}

treated = merged[merged['ever_treated'] == 1]
never   = merged[merged['ever_treated'] == 0]

from scipy import stats

rows = []
for var, label in BALANCE_VARS.items():
    t_vals = treated[var].dropna()
    n_vals = never[var].dropna()

    t_mean = t_vals.mean()
    t_sd   = t_vals.std()
    n_mean = n_vals.mean()
    n_sd   = n_vals.std()
    diff   = t_mean - n_mean

    # Two-sample t-test
    _, p_val = stats.ttest_ind(t_vals, n_vals, equal_var=False)

    stars = ''
    if p_val < 0.01:   stars = '***'
    elif p_val < 0.05: stars = '**'
    elif p_val < 0.10: stars = '*'

    rows.append({
        'Variable':           label,
        'Treated Mean':       t_mean,
        'Treated SD':         t_sd,
        'Never-Treated Mean': n_mean,
        'Never-Treated SD':   n_sd,
        'Difference':         diff,
        'Stars':              stars,
        'p-value':            p_val,
    })

balance = pd.DataFrame(rows)
balance.to_csv("balance_table.csv", index=False)

#%%

# Print balance table
print("\nBalance Table: Treated vs Never-Treated Census Tracts")
print("ACS 2006-2010 5-Year Estimates")
print("=" * 75)
print(f"{'Variable':<35} {'Treated':>10} {'Never-Treated':>14} {'Diff':>10}")
print(f"{'':35} {'Mean (SD)':>10} {'Mean (SD)':>14} {'':>10}")
print("-" * 75)

for _, row in balance.iterrows():
    var_key = [k for k, v in BALANCE_VARS.items() if v == row['Variable']][0]
    t_str = f"{fmt(row['Treated Mean'], var_key)} ({fmt(row['Treated SD'], var_key)})"
    n_str = f"{fmt(row['Never-Treated Mean'], var_key)} ({fmt(row['Never-Treated SD'], var_key)})"
    d_str = f"{fmt(row['Difference'], var_key)}{row['Stars']}"
    print(f"{row['Variable']:<35} {t_str:>10} {n_str:>14} {d_str:>10}")

print("-" * 75)
print("* p<0.10   ** p<0.05   *** p<0.01")
