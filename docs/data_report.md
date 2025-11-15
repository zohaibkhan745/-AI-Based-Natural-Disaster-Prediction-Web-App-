# Meteostat Weather Data Export

## Overview

- **Source:** [Meteostat Daily API](https://meteostat.net/) via the official Python client (v1.6.7 from `requirements.txt`).
- **Extraction Script:** `python -m code.fetch_meteostat_weather --combine`
- **Default Window:** 2018-01-01 through the current day at run time (2025-11-15 for this export).
- **Output Directory:** `data/raw/`

Running the script downloads per-location CSVs and, when `--combine` is set, a merged `weather_combined.csv`. Each file follows the same schema and includes station metadata columns for traceability.

## Locations

| Key         | Human Name                             | Latitude   | Longitude  | Elevation (m) | Notes                                                                                          |
| ----------- | -------------------------------------- | ---------- | ---------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `swat`      | Swat District, Khyber Pakhtunkhwa      | 34.8091° N | 72.3617° E | 980           | Represents Mingora valley conditions; Meteostat station IDs 41523/41501 supply most rows.      |
| `upper_dir` | Upper Dir District, Khyber Pakhtunkhwa | 35.3350° N | 71.8760° E | 1420          | Captures Hindu Kush foothills climate; Meteostat station IDs 41508/41505 dominate the dataset. |

> Coordinates/elevations are approximations based on district centroids; adjust in `code/fetch_meteostat_weather.py` if higher-fidelity station metadata becomes available.

## Files Created

| File                                                   | Rows  | Description                                                                                        |
| ------------------------------------------------------ | ----- | -------------------------------------------------------------------------------------------------- |
| `data/raw/weather_swat_2018-01-01_2025-11-15.csv`      | 1,042 | Daily summary for Swat; Meteostat did not provide coverage before 2021, hence warnings in the log. |
| `data/raw/weather_upper_dir_2018-01-01_2025-11-15.csv` | 2,872 | Daily summary for Upper Dir; mostly complete for 2018 onward.                                      |
| `data/raw/weather_combined.csv`                        | 3,914 | Concatenation of the above with identical schema, sorted by `date`.                                |

All files share the following columns (after resetting the index returned by Meteostat):

| Column                                                                  | Meaning                                                           |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `date`                                                                  | ISO date of measurement (local time).                             |
| `tavg`, `tmin`, `tmax`                                                  | Daily average/min/max temperature in °C.                          |
| `prcp`, `snow`                                                          | Precipitation depth (mm) and snowfall (mm).                       |
| `wdir`, `wspd`, `wpgt`                                                  | Mean wind direction (°), mean speed (km/h), and peak gust (km/h). |
| `pres`                                                                  | Air pressure (hPa).                                               |
| `tsun`                                                                  | Sunshine duration (minutes).                                      |
| `location_key`, `location_name`, `latitude`, `longitude`, `elevation_m` | Metadata injected by the script for downstream joins/filters.     |

Missing measurements are left blank by Meteostat and therefore appear as empty cells in the CSVs. Pandas will interpret them as `NaN` automatically when loading the files.

## Reproducibility Checklist

1. Activate the project virtual environment: `source .venv/Scripts/activate` (PowerShell automatically activates via `.venv`).
2. From the repo root, run `python -m code.fetch_meteostat_weather --start-date YYYY-MM-DD --end-date YYYY-MM-DD --combine`.
3. Inspect `data/raw/` for fresh CSVs and review the INFO/Warning log output for coverage issues.

## NASA POWER Data Integration

To address missing values in the Meteostat dataset, NASA POWER API data has been integrated as a secondary source.

### Fetching NASA POWER Data

```bash
python -m code.fetch_nasa_power --combine
```

This fetches the following parameters from NASA POWER:
- Temperature (T2M, T2M_MIN, T2M_MAX)
- Precipitation (PRECTOTCORR)
- Wind speed (WS2M, WS2M_MAX)
- Humidity (RH2M) - **Not available in Meteostat**
- Pressure (PS)
- Solar radiation (ALLSKY_SFC_SW_DWN) - **Not available in Meteostat**

### Merging Datasets

The `merge_weather_data.py` script combines Meteostat and NASA POWER data:

```bash
python -m code.merge_weather_data
```

**Merge Strategy:**
1. Meteostat data is used as the primary source
2. Missing values in Meteostat are filled with NASA POWER data
3. Humidity and solar radiation columns are added from NASA POWER

**Results:**
- **Before merge:** 26-28% missing values in key weather features
- **After merge:** 0% missing values for tavg, tmin, tmax, prcp, wspd, wpgt, pres
- **Added columns:** humidity, solar_radiation
- **Output:** `data/processed/weather_merged.csv` (3,914 rows × 18 columns)

| Column              | Missing Before | Missing After | Filled |
| ------------------- | -------------- | ------------- | ------ |
| tavg                | 1,047 (26.8%)  | 0 (0%)        | 100%   |
| tmin                | 1,080 (27.6%)  | 0 (0%)        | 100%   |
| tmax                | 1,047 (26.8%)  | 0 (0%)        | 100%   |
| prcp                | 1,063 (27.2%)  | 0 (0%)        | 100%   |
| wspd                | 1,109 (28.3%)  | 0 (0%)        | 100%   |
| wpgt                | 3,914 (100%)   | 0 (0%)        | 100%   |
| pres                | 1,115 (28.5%)  | 0 (0%)        | 100%   |
| humidity (new)      | N/A            | 0 (0%)        | N/A    |
| solar_radiation (new) | N/A          | 0 (0%)        | N/A    |

### Final Dataset for ML

The merged dataset (`data/processed/weather_merged.csv`) is ready for machine learning with:

**Input Features (X):**
- date
- tavg, tmin, tmax (temperature in °C)
- prcp (rainfall in mm)
- snow (snowfall in mm - mostly missing)
- wspd, wpgt (wind speed and gust in km/h)
- pres (pressure in hPa)
- humidity (relative humidity in %, from NASA)
- solar_radiation (solar radiation in W/m², from NASA)
- location (encoded as location_key)

**Output (y) - To be added:**
- flood_event (0/1) - Manual labeling from NDMA reports required

## Next Steps

- **Label flood events:** Use NDMA reports to create flood_event target variable (1 if flood reported, 0 otherwise)
- Pull additional Meteostat stations nearer to flood-prone tehsils if higher spatial resolution is required.
- Enrich files with location-specific columns (e.g., district code, population) to support modeling.
- Train ML model using the merged dataset with complete weather features.
