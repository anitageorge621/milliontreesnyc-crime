#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: anitag
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_panel.py
MillionTreesNYC — Panel Construction
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import warnings
from io import BytesIO
warnings.filterwarnings('ignore')

#%%


# Store the crime outcomes to be looked at in a list
CRIME_TYPES = [
    "FELONY ASSAULT", "ROBBERY", "BURGLARY",
    "PETIT LARCENY", "GRAND LARCENY",
    "GRAND LARCENY OF MOTOR VEHICLE"
]

YEAR_START   = 2006
YEAR_END     = 2019
REL_YEAR_MIN = -4   # bin all pre-periods beyond -4 together
REL_YEAR_MAX =  8   # bin all post-periods beyond +8 together

# Label boroughs
BOROUGH_MAP = {
    '005': 'Bronx', '047': 'Brooklyn', '061': 'Manhattan',
    '081': 'Queens', '085': 'Staten Island'
}

# Function to standardize all GEOID variables
def std_geoid(s):
    # Zero-pad GEOID to 11 characters for consistent merging
    return s.astype(str).str.strip().str.zfill(11)

# Function to create proper bin windows for the event study
def bin_rel_year(x):
    # Bin relative year to the event study window
    if pd.isna(x):          return np.nan
    elif x <= REL_YEAR_MIN: return REL_YEAR_MIN
    elif x >= REL_YEAR_MAX: return REL_YEAR_MAX
    else:                   return int(x)

#%%

# 1. Fetching NYC census tracts from Census Bureau API

# Use your own API key stored in apikey.txt
with open('apikey.txt', 'r') as f:
    CENSUS_API_KEY = f.read().strip()

# Create an empty list to store tracts 
nyc_tracts_list = []

# Loop through to get all census tracts
for county_fips, borough_name in BOROUGH_MAP.items():
    # Query TIGERweb API for 2010 census tract boundaries for each borough
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Census2010/MapServer/14/query"
    )
    params = {
        "where":     f"STATE='36' AND COUNTY='{county_fips}'",
        "outFields": "GEOID,AREALAND,AREAWATER",
        "outSR":     "4326",    # WGS84 lat/lon
        "f":         "geojson",
        "token":     CENSUS_API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    gdf = gpd.read_file(BytesIO(response.content))
    gdf['borough'] = borough_name
    nyc_tracts_list.append(gdf)

nyc_tracts = pd.concat(nyc_tracts_list, ignore_index=True)
nyc_tracts = nyc_tracts.to_crs(epsg=4326)
nyc_tracts['GEOID'] = std_geoid(nyc_tracts['GEOID'])

# Drop water-only tracts (no land area = no residents, no crime)
water_geoids = set(nyc_tracts[nyc_tracts['AREALAND'] == 0]['GEOID'])
nyc_tracts   = nyc_tracts[~nyc_tracts['GEOID'].isin(water_geoids)].copy()

print(f"   {len(nyc_tracts)} tracts loaded ({len(water_geoids)} water tracts removed)")

#%%

# 2. Tree Planting Data

# Load tree planting data
trees = gpd.read_file("MTNYC_PlantedBlocks_20150930_20251107")

# Extract numeric planting year from season string (e.g. "07-08" → 2007)
trees['planting_year'] = (
    trees['plnt_seas'].str.extract(r'(\d+)').astype(int) + 2000
)

# Keep only plantings from 2007 onward (program start year)
trees = trees[trees['planting_year'] >= 2007].copy()
trees = trees.to_crs(epsg=4326)

# Reproject to UTM Zone 18N for accurate centroid calculation (meters)
trees_proj  = trees.to_crs(epsg=32618)
tracts_proj = nyc_tracts.to_crs(epsg=32618)

# Use centroid of each planted block for the spatial join
trees_proj['geometry'] = trees_proj.geometry.centroid

# Assign each planted block to the census tract it falls within
trees_joined = gpd.sjoin(
    trees_proj[['planting_year', 'geometry']],
    tracts_proj[['GEOID', 'geometry']],
    how='left', predicate='intersects'
)

# Snap any unmatched blocks to the nearest tract
unmatched = trees_joined['GEOID'].isna()
if unmatched.sum() > 0:
    nearest = gpd.sjoin_nearest(
        trees_proj[unmatched][['planting_year', 'geometry']],
        tracts_proj[['GEOID', 'geometry']]
    )
    trees_joined.loc[unmatched, 'GEOID'] = nearest['GEOID'].values

# One row per tract: the first year any tree was planted there
treated_tracts = (
    trees_joined.groupby('GEOID')['planting_year']
    .min().reset_index()
    .rename(columns={'planting_year': 'first_planting_year'})
)
treated_tracts['ever_treated'] = 1

print(f"   {len(treated_tracts)} treated tracts identified")

#%%

# 3. Crime Data

# Load crime data
crime_raw = pd.read_csv("NYPD_Complaint_Data_(2006-19).csv")

# Format date of crime
crime_raw['date'] = pd.to_datetime(
    crime_raw['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce'
)

crime_raw['year'] = crime_raw['date'].dt.year

# Filter to target crime types, study years, and valid NYC coordinates
crime_clean = crime_raw[
    crime_raw['OFNS_DESC'].isin(CRIME_TYPES) &
    crime_raw['year'].between(YEAR_START, YEAR_END) &
    crime_raw['Latitude'].notna() &
    crime_raw['Longitude'].notna() &
    crime_raw['Latitude'].between(40, 41) &
    crime_raw['Longitude'].between(-75, -73)
].copy()

print(f"   {len(crime_clean):,} crime records retained")

# Convert to GeoDataFrame and assign each crime to a census tract
crime_gdf = gpd.GeoDataFrame(
    crime_clean[['year', 'OFNS_DESC']],
    geometry=gpd.points_from_xy(crime_clean['Longitude'], crime_clean['Latitude']),
    crs='EPSG:4326'
)

# Spatial join to connect each crime to the census tract it occurred in 
crime_geocoded = gpd.sjoin(
    crime_gdf, nyc_tracts[['GEOID', 'geometry']],
    how='left', predicate='intersects'
)
crime_geocoded = crime_geocoded[crime_geocoded['GEOID'].notna()].copy()
crime_geocoded['GEOID'] = std_geoid(crime_geocoded['GEOID'])

print(f"   {len(crime_geocoded):,} crimes geocoded to tracts")

#%%

# 4. Build Annual Panel

# Keep only tracts with a unique GEOID
keep_geoids = nyc_tracts['GEOID'].unique().tolist()

# Aggregate to tract × year × crime type counts
crime_annual = (
    crime_geocoded[crime_geocoded['GEOID'].isin(keep_geoids)]
    .groupby(['GEOID', 'year', 'OFNS_DESC'])
    .size().reset_index(name='crime_count')
)

# Build complete balanced grid — ensures zero-crime tract-years are explicit
full_index = pd.MultiIndex.from_product(
    [keep_geoids, range(YEAR_START, YEAR_END + 1), CRIME_TYPES],
    names=['GEOID', 'year', 'OFNS_DESC']
).to_frame(index=False)

# Merge the full index grid with annual crime list
panel = full_index.merge(crime_annual, on=['GEOID', 'year', 'OFNS_DESC'], how='left')

# Explicitly label zero crime counts
panel['crime_count'] = panel['crime_count'].fillna(0).astype(int)

#%%

# 5. Merge Treatment Variables 


panel = panel.merge(nyc_tracts[['GEOID', 'borough']], on='GEOID', how='left')
panel = panel.merge(treated_tracts, on='GEOID', how='left')
panel['ever_treated'] = panel['ever_treated'].fillna(0).astype(int)

# Relative year: years since first treatment (NaN for never-treated)
panel['rel_year'] = panel['year'] - panel['first_planting_year']
panel['rel_year_binned'] = panel['rel_year'].apply(bin_rel_year)

# Cohort: first treatment year (0 for never-treated), used by pyfixest did2s
panel['cohort'] = panel['first_planting_year'].fillna(0).astype(int)

# Borough × year fixed effect identifier
panel['borough_year'] = panel['borough'] + '_' + panel['year'].astype(str)

# Log crime count for descriptive visualizations
panel['log_crime'] = np.log1p(panel['crime_count'])

print(f"   {panel['GEOID'].nunique()} tracts × {panel['year'].nunique()} years × "
      f"{panel['OFNS_DESC'].nunique()} crime types")
print(f"   Ever-treated:  {panel[panel['ever_treated']==1]['GEOID'].nunique()}")
print(f"   Never-treated: {panel[panel['ever_treated']==0]['GEOID'].nunique()}")
print(f"   Total rows: {len(panel):,}")

#%%

# 6. Save tract-level summary for use in visualizations

# Count total blocks planted per tract for the intensity map
blocks_per_tract = (
    trees_joined.groupby('GEOID').size()
    .reset_index(name='blocks_planted')
)

tract_summary = (
    panel.groupby('GEOID')
    .agg(ever_treated        = ('ever_treated',        'first'),
         first_planting_year = ('first_planting_year', 'first'),
         borough             = ('borough',             'first'))
    .reset_index()
)
tract_summary = tract_summary.merge(blocks_per_tract, on='GEOID', how='left')
tract_summary['blocks_planted'] = tract_summary['blocks_planted'].fillna(0)
tract_summary.to_csv("tract_summary.csv", index=False)


#%%

# 7. Save panel data
panel.to_csv("regression_panel_annual.csv", index=False)



