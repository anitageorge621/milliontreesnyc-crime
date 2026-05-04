## MillionTreesNYC and Crime

This project estimates the causal effect of street tree planting on crime incidence across New York City census tracts. Using data from the MillionTreesNYC program — a city-wide initiative that planted over one million trees between 2007 and 2015 — the analysis asks whether the arrival of street trees in a neighborhood reduces the likelihood that crime occurs there.

The empirical strategy exploits the staggered rollout of tree planting across census tracts to estimate a difference-in-differences event study. Because different tracts received their first trees in different years, we can compare crime trends in newly treated tracts against tracts not yet treated, before and after treatment. The estimator follows Gardner (2021), a two-stage DiD approach robust to treatment effect heterogeneity across cohorts. The outcome is an extensive margin indicator — whether any crime of a given type occurred in a tract-year — estimated separately for six felony crime types: felony assault, robbery, burglary, petit larceny, grand larceny, and grand larceny of motor vehicle.

---

## Data

The raw data files are not included in this repository due to file size. Download each file and place it in the same folder as the scripts before running.

### 1. MillionTreesNYC Planted Blocks

- **Source:** NYC Open Data
- **URL:** <https://data.cityofnewyork.us/Environment/MillionTreesNYC-Planted-Trees/k6hx-ej4w>
- **Download:** Click **Export → Shapefile**. Unzip and place the folder `MTNYC_PlantedBlocks_20150930_20251107/` in the repo directory.

### 2. NYPD Complaint Data (Historic)

- **Source:** NYC Open Data
- **URL:** <https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i>
- **Download:** Click **Export → CSV**. Rename the file to `NYPD_Complaint_Data_(2006-19).csv` and place it in the repo directory.

### 3. Census Tract Boundaries and ACS Data

No manual download needed. Census tract boundaries are fetched automatically from the U.S. Census Bureau TIGERweb API, and ACS 2006-2010 5-year estimates are fetched from the Census Bureau data API. Both require a free Census API key, available at <https://api.census.gov/data/key_signup.html>. Once you have a key, create a file called `apikey.txt` in the repo directory and paste your key inside.

---

## Scripts

Run the scripts in order from the repo directory.

### `01_build_panel.py`

Loads the MillionTreesNYC shapefile and NYPD crime data, geocodes both datasets to 2010 census tracts via spatial join, and constructs a balanced annual panel at the tract × year × crime type level covering 2006–2019. Defines treatment variables including each tract's first planting year, ever-treated indicator, relative year, and cohort. Outputs `regression_panel_annual.csv` and `tract_summary.csv`.

### `02_extensive_margin.py`

Loads the annual panel and estimates the Gardner (2021) two-stage DiD event study for each of the six crime types. Stage 1 absorbs tract and year fixed effects using untreated and not-yet-treated observations only. Stage 2 regresses the residualized outcome on event-time dummies with cluster-robust standard errors at the tract level. The outcome is a binary indicator for whether any crime occurred in a given tract-year. Outputs `event_study_results.csv` and prints a coefficient table to the console.

### `03_event_study_plot.py`

Loads `event_study_results.csv` and produces a 2×3 grid of event study plots with 95% confidence intervals, one panel per crime type. The x-axis shows years relative to first tree planting and the y-axis shows the estimated change in the probability of any crime occurring. Outputs `event_study.png`.

### `04_visualizations.py`

Produces four descriptive figures using the panel and tract summary data: (1) a choropleth map of treatment timing by census tract, (2) a map of tree planting intensity by tract, (3) a stacked bar chart of the planting rollout by borough and year, and (4) box plots of the annual crime count distribution by borough for each crime type. Outputs `map_treatment_timing.png`, `map_planting_intensity.png`, `rollout_by_borough.png`, and `crime_distributions.png`.

### `05_balance_table.py`

Fetches ACS 2006-2010 5-year estimates from the Census API and constructs a balance table comparing treated and never-treated tracts on median household income, poverty rate, unemployment rate, total population, race/ethnicity shares, and educational attainment. Reports means, standard deviations, and tests for differences using a two-sample t-test. Outputs `balance_table.csv` and `balance_table.png`.

---

## Output Files

| File | Description |
|---|---|
| `event_study.png` | Event study plots for all six crime types |
| `map_treatment_timing.png` | Choropleth map of first planting year by census tract |
| `map_planting_intensity.png` | Map of total blocks planted per tract |
| `rollout_by_borough.png` | Stacked bar chart of planting rollout by borough and year |
| `crime_distributions.png` | Box plots of annual crime counts by borough and crime type |
| `balance_table.png` | Balance table comparing treated and never-treated tracts |

---

## Results

The results vary across crime types. For grand larceny of motor vehicle, the estimates show statistically significant negative effects in the first two to three years following treatment, consistent with the hypothesis that increased street activity deters opportunistic vehicle theft. This is the clearest evidence of a deterrent effect and aligns with the "eyes on the street" mechanism.

For the remaining five crime types (felony assault, robbery, burglary, petit larceny, and grand larceny), post-treatment coefficients are generally positive and grow over time. Since the Gardner (2021) estimator absorbs all time-invariant baseline differences through unit fixed effects in stage one, these are causal estimates rather than artifacts of selection into treatment. The positive effects likely reflect two mechanisms: increased foot traffic in newly greened neighborhoods raises the number of potential victims and witnesses, and greater community engagement in these areas may increase crime reporting rates relative to untreated tracts. These findings suggest that street trees have a complex and heterogeneous effect, depending on the crime type
