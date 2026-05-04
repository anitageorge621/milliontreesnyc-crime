#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: anitag
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_visualizations.py
MillionTreesNYC — Descriptive Visualizations
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import warnings
from io import BytesIO
warnings.filterwarnings('ignore')

# Census API Key
with open('apikey.txt', 'r') as f:
    CENSUS_API_KEY = f.read().strip()

# Settings 
BOROUGH_MAP = {
    '005': 'Bronx', '047': 'Brooklyn', '061': 'Manhattan',
    '081': 'Queens', '085': 'Staten Island'
}
BOROUGH_ORDER  = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']
BOROUGH_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

CRIME_LABELS = {
    "FELONY ASSAULT":                 "Felony Assault",
    "ROBBERY":                        "Robbery",
    "BURGLARY":                       "Burglary",
    "PETIT LARCENY":                  "Petit Larceny",
    "GRAND LARCENY":                  "Grand Larceny",
    "GRAND LARCENY OF MOTOR VEHICLE": "Grand Larceny\nof Motor Vehicle",
}

def std_geoid(s):
    return s.astype(str).str.strip().str.zfill(11)

#%%

# 1. Load Data

# Load panel data
panel      = pd.read_csv("regression_panel_annual.csv")
panel['first_planting_year'] = pd.to_numeric(
    panel['first_planting_year'], errors='coerce'
)

# Load tract summary (has ever_treated, first_planting_year, borough, blocks_planted)
tract_info = pd.read_csv("tract_summary.csv")
tract_info['GEOID'] = std_geoid(tract_info['GEOID'])

# Fetch census tract boundaries (needed for map geometries)
nyc_tracts_list = []
for county_fips, borough_name in BOROUGH_MAP.items():
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Census2010/MapServer/14/query"
    )
    params = {
        "where":     f"STATE='36' AND COUNTY='{county_fips}'",
        "outFields": "GEOID,AREALAND", "outSR": "4326", "f": "geojson",
        "token":     CENSUS_API_KEY,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    gdf = gpd.read_file(BytesIO(r.content))
    gdf['borough'] = borough_name
    nyc_tracts_list.append(gdf)

nyc_tracts = pd.concat(nyc_tracts_list, ignore_index=True)
nyc_tracts = nyc_tracts.to_crs(epsg=4326)
nyc_tracts['GEOID'] = std_geoid(nyc_tracts['GEOID'])

# Drop water-only tracts
water_geoids = set(nyc_tracts[nyc_tracts['AREALAND'] == 0]['GEOID'])
nyc_tracts   = nyc_tracts[~nyc_tracts['GEOID'].isin(water_geoids)].copy()

# Merge tract_info (treatment status + blocks_planted) onto shapefile for mapping
map_gdf = nyc_tracts.merge(tract_info, on='GEOID', how='left')
map_gdf['blocks_planted'] = map_gdf['blocks_planted'].fillna(0)
map_gdf['ever_treated']   = map_gdf['ever_treated'].fillna(0)

#%%

# Figure 1: Treatment Timing Choropleth

fig, ax = plt.subplots(figsize=(9, 10))

# Never-treated tracts in grey
map_gdf[map_gdf['ever_treated'] == 0].plot(
    ax=ax, color='#e0e0e0', linewidth=0.1, edgecolor='white'
)
# Treated tracts colored by first planting year
map_gdf[map_gdf['ever_treated'] == 1].plot(
    column='first_planting_year', ax=ax, cmap='YlGn',
    linewidth=0.1, edgecolor='white', legend=True,
    legend_kwds={'label': 'First planting year',
                 'orientation': 'horizontal', 'shrink': 0.6, 'pad': 0.02}
)
never_patch = mpatches.Patch(color='#e0e0e0', label='Never treated')
ax.legend(handles=[never_patch], loc='lower left', fontsize=9, frameon=False)
ax.set_title("MillionTreesNYC Treatment Timing by Census Tract",
             fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("map_treatment_timing.png", dpi=150, bbox_inches='tight')
plt.close()

#%%

# ── Figure 2: Planting Intensity Map ─────────────────────────────────────────
print("2. Drawing planting intensity map...")

fig, ax = plt.subplots(figsize=(9, 10))

map_gdf[map_gdf['blocks_planted'] == 0].plot(
    ax=ax, color='#e0e0e0', linewidth=0.1, edgecolor='white'
)
int_gdf = map_gdf[map_gdf['blocks_planted'] > 0].copy()
int_gdf['log_blocks'] = np.log1p(int_gdf['blocks_planted'])
int_gdf.plot(
    column='log_blocks', ax=ax, cmap='Blues',
    linewidth=0.1, edgecolor='white', legend=True,
    legend_kwds={'label': 'Blocks planted (log scale)',
                 'orientation': 'horizontal', 'shrink': 0.6, 'pad': 0.02}
)
no_tree_patch = mpatches.Patch(color='#e0e0e0', label='No trees planted')
ax.legend(handles=[no_tree_patch], loc='lower left', fontsize=9, frameon=False)
ax.set_title("MillionTreesNYC Planting Intensity by Census Tract",
             fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("map_planting_intensity.png", dpi=150, bbox_inches='tight')
plt.close()


#%%

# ── Figure 3: Rollout by Borough and Year ────────────────────────────────────
print("3. Drawing rollout bar chart...")

rollout = (
    panel[panel['ever_treated'] == 1]
    .groupby(['borough', 'first_planting_year'])['GEOID']
    .nunique().reset_index(name='tracts_treated')
    .rename(columns={'first_planting_year': 'year'})
    .dropna(subset=['year'])
)
rollout['year'] = rollout['year'].astype(int)

rollout_pivot = (
    rollout.pivot_table(index='year', columns='borough',
                        values='tracts_treated', fill_value=0)
    .reindex(columns=BOROUGH_ORDER, fill_value=0)
)

fig, ax = plt.subplots(figsize=(11, 5))
rollout_pivot.plot(kind='bar', stacked=True, ax=ax,
                   color=BOROUGH_COLORS, width=0.7,
                   edgecolor='white', linewidth=0.5)
ax.set_title("Tree Planting Rollout: Newly Treated Tracts by Borough and Year",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number of tracts first treated", fontsize=10)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.legend(title='Borough', bbox_to_anchor=(1.01, 1),
          loc='upper left', fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("rollout_by_borough.png", dpi=150, bbox_inches='tight')
plt.close()


#%%

# ── Figure 4: Crime Distributions by Borough ─────────────────────────────────
print("4. Drawing crime distribution box plots...")

crime_types = sorted(panel['OFNS_DESC'].unique())

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, crime in enumerate(crime_types):
    ax     = axes[i]
    df     = panel[panel['OFNS_DESC'] == crime]
    groups = [df[df['borough'] == b]['crime_count'].values
              for b in BOROUGH_ORDER]

    bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))
    for patch, color in zip(bp['boxes'], BOROUGH_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(CRIME_LABELS.get(crime, crime), fontsize=10, fontweight='bold')
    ax.set_xticks(range(1, len(BOROUGH_ORDER) + 1))
    ax.set_xticklabels([b.replace(' ', '\n') for b in BOROUGH_ORDER], fontsize=8)
    ax.set_ylabel("Annual crime count per tract", fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    "Distribution of Annual Crime Counts by Borough and Crime Type\n"
    "(tract-year observations, outliers excluded)",
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig("crime_distributions.png", dpi=150, bbox_inches='tight')
plt.close()


