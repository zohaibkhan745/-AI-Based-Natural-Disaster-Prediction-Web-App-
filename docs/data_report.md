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

## Next Steps

- Pull additional Meteostat stations nearer to flood-prone tehsils if higher spatial resolution is required.
- Enrich files with location-specific columns (e.g., district code, population) to support modeling.
- Promote the combined dataset into `data/processed/` after QA (outlier handling, missing-value strategies, rolling aggregates, etc.).
