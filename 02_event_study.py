#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: anitag
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_event_studyn.py
MillionTreesNYC — Event Study Estimation
==============================================
Estimates the causal effect of MillionTreesNYC tree plantings on the
probability that any crime occurs in a census tract-year, using the
Gardner (2021) two-stage DiD estimator via pyfixest.


"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%%

# 1. Load Panel 

panel = pd.read_csv("regression_panel_annual.csv")
panel['first_planting_year'] = pd.to_numeric(
    panel['first_planting_year'], errors='coerce'
)

# Extensive margin outcome: 1 if any crime occurred in this tract-year, else 0
panel['any_crime'] = (panel['crime_count'] > 0).astype(int)

print(f"   {len(panel):,} rows | {panel['OFNS_DESC'].nunique()} crime types")

#%%

# 2. Estimation: Running Gardner (2021) DiD2s for each crime type

from sklearn.linear_model import LinearRegression

results_list = []

for crime in sorted(panel['OFNS_DESC'].unique()):

    df = panel[panel['OFNS_DESC'] == crime].copy().reset_index(drop=True)
    df['any_crime'] = (df['crime_count'] > 0).astype(int)
    df['rel_year_binned'] = df['rel_year_binned'].fillna(-99).astype(int)
    df['tract_id'] = pd.factorize(df['GEOID'])[0]

    # Stage 1: absorb unit + year FEs using untreated observations only
    # "Untreated" = never treated OR not yet treated at time t
    untreated = df[(df['ever_treated'] == 0) | 
                   (df['rel_year'] < 0) | 
                   (df['rel_year'].isna())].copy()

    # Demean by tract and year using untreated obs only
    untreated['tract_mean'] = untreated.groupby('tract_id')['any_crime'].transform('mean')
    untreated['year_mean']  = untreated.groupby('year')['any_crime'].transform('mean')
    untreated['grand_mean'] = untreated['any_crime'].mean()

    # Build FE lookup tables from untreated obs
    tract_fe = (untreated.groupby('tract_id')['any_crime'].mean()
                - untreated['any_crime'].mean())
    year_fe  = (untreated.groupby('year')['any_crime'].mean()
                - untreated['any_crime'].mean())
    grand_mean = untreated['any_crime'].mean()

    # ── Stage 2: partial out FEs from full sample, regress on event dummies ──
    df['tract_fe'] = df['tract_id'].map(tract_fe).fillna(0)
    df['year_fe']  = df['year'].map(year_fe).fillna(0)
    df['y_resid']  = df['any_crime'] - df['tract_fe'] - df['year_fe'] - grand_mean

    # Run event study on treated obs only (exclude never-treated and placeholder)
    treated_obs = df[df['rel_year_binned'] != -99].copy()

    # Event time dummies (excluding ref period -1)
    rel_years = sorted([r for r in treated_obs['rel_year_binned'].unique() if r != -1])
    for r in rel_years:
        treated_obs[f'd_{r}'] = (treated_obs['rel_year_binned'] == r).astype(int)

    dummy_cols = [f'd_{r}' for r in rel_years]
    X = treated_obs[dummy_cols].values
    y = treated_obs['y_resid'].values

    # OLS with clustered SEs (by tract)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta

    # Cluster-robust variance
    tracts  = treated_obs['tract_id'].values
    meat    = np.zeros((X.shape[1], X.shape[1]))
    for t in np.unique(tracts):
        idx  = tracts == t
        Xg   = X[idx]
        eg   = resid[idx]
        meat += Xg.T @ np.outer(eg, eg) @ Xg
    n, k   = X.shape
    n_clust = len(np.unique(tracts))
    scale  = (n_clust / (n_clust - 1)) * ((n - 1) / (n - k))
    V      = scale * XtX_inv @ meat @ XtX_inv
    se     = np.sqrt(np.diag(V))

    # Build results dataframe
    coef_df = pd.DataFrame({
        'rel_year':   rel_years,
        'coef':       beta,
        'se':         se,
        'ci_low':     beta - 1.96 * se,
        'ci_high':    beta + 1.96 * se,
        'crime_type': crime
    })
    # Add reference period row (normalized to 0)
    ref_row = pd.DataFrame([{
        'rel_year': -1, 'coef': 0.0, 'se': 0.0,
        'ci_low': 0.0, 'ci_high': 0.0, 'crime_type': crime
    }])
    coef_df = pd.concat([coef_df, ref_row]).sort_values('rel_year').reset_index(drop=True)
    results_list.append(coef_df[['crime_type','rel_year','coef','se','ci_low','ci_high']])
    print(f"   ✓ {crime}")


#%%


# Save results
results = pd.concat(results_list, ignore_index=True)
results.to_csv("event_study_results.csv", index=False)

    
#%%

# 3. Print Summary Table 

# Show post-treatment coefficients (rel_year >= 0) for each crime type
print("\nEvent Study Results — Post-Treatment Coefficients")
print("=" * 65)

for crime in sorted(results['crime_type'].unique()):
    df = results[(results['crime_type'] == crime) & (results['rel_year'] >= 0)]
    print(f"\n{crime}")
    print(f"  {'Rel. Year':<12} {'Coef':>8} {'SE':>8} {'CI Low':>8} {'CI High':>8}")
    print(f"  {'-'*52}")
    for _, row in df.iterrows():
        sig = "*" if (row['ci_low'] > 0 or row['ci_high'] < 0) else ""
        print(f"  {int(row['rel_year']):<12} {row['coef']:>8.4f} {row['se']:>8.4f} "
              f"{row['ci_low']:>8.4f} {row['ci_high']:>8.4f} {sig}")

print("\n* = 95% CI excludes zero")    

