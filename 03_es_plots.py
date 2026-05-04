#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:25:35 2026

@author: anitag
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_event_study_plot.py
MillionTreesNYC — Event Study Plots

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#%%

# 1. Load Results 

results    = pd.read_csv("event_study_results.csv")
crime_types = sorted(results['crime_type'].unique())

# Clean display labels for each crime type
LABELS = {
    "FELONY ASSAULT":                 "Felony Assault",
    "ROBBERY":                        "Robbery",
    "BURGLARY":                       "Burglary",
    "PETIT LARCENY":                  "Petit Larceny",
    "GRAND LARCENY":                  "Grand Larceny",
    "GRAND LARCENY OF MOTOR VEHICLE": "Grand Larceny\nof Motor Vehicle",
}

#%%

# 2. Plot

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

for i, crime in enumerate(crime_types):
    ax = axes[i]
    df = results[results['crime_type'] == crime].sort_values('rel_year')

    # Shade pre-treatment period in light grey
    ax.axvspan(df['rel_year'].min() - 0.5, -0.5,
               color='lightgrey', alpha=0.3, label='Pre-treatment')

    # Reference lines
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='firebrick', linewidth=0.8, linestyle='--',
               label='Treatment year')

    # 95% confidence interval shading
    ax.fill_between(df['rel_year'], df['ci_low'], df['ci_high'],
                    alpha=0.2, color='steelblue')

    # Point estimates
    ax.plot(df['rel_year'], df['coef'],
            color='steelblue', linewidth=1.8,
            marker='o', markersize=4, label='Estimate')

    ax.set_title(LABELS.get(crime, crime), fontsize=11, fontweight='bold')
    ax.set_xlabel("Years since first tree planting", fontsize=9)
    ax.set_ylabel("Δ Pr(any crime)", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Effect of MillionTreesNYC on Crime Incidence\n"
    "Gardner (2021) Two-Stage DiD, 95% Confidence Intervals",
    fontsize=13, fontweight='bold', y=1.01
)

plt.tight_layout()
plt.savefig("event_study.png", dpi=150, bbox_inches='tight')
plt.close()
