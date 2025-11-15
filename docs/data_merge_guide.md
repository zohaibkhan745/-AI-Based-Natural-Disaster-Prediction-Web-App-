# Weather Data Merging Guide

This guide explains how to merge Meteostat and NASA POWER weather data to create a complete dataset for machine learning.

## Problem

Meteostat data has significant missing values:
- 26-28% missing values for key weather features (temperature, pressure, rainfall, wind)
- Some columns completely missing (humidity, solar radiation)

## Solution

Use NASA POWER API as a secondary data source to fill missing values and add new features.

## Step-by-Step Process

### 1. Fetch Meteostat Data

```bash
python -m code.fetch_meteostat_weather --combine
```

**Output:**
- `data/raw/weather_swat_*.csv`
- `data/raw/weather_upper_dir_*.csv`
- `data/raw/weather_combined.csv` (combined file)

**Coverage:** 2018-01-01 to present, but with significant gaps

### 2. Fetch NASA POWER Data

```bash
python -m code.fetch_nasa_power --combine
```

**Output:**
- `data/raw/nasa_power_swat_*.csv`
- `data/raw/nasa_power_upper_dir_*.csv`
- `data/raw/nasa_power_combined.csv` (combined file)

**Coverage:** Same date range as Meteostat, with complete coverage

**Note:** NASA POWER API may be blocked in some network environments. In such cases:
- Use the pre-generated NASA POWER data in `data/raw/`
- Or run the script from a network with access to NASA's servers
- Or generate synthetic NASA data for testing (see Advanced section below)

### 3. Merge Both Datasets

```bash
python -m code.merge_weather_data
```

**Output:**
- `data/processed/weather_merged.csv` (merged dataset with 0% missing values)

**Merge Logic:**
1. Load both Meteostat and NASA POWER data
2. Align by date and location_key
3. For each weather parameter:
   - Use Meteostat value if available (primary source)
   - Fill missing Meteostat values with NASA POWER data (secondary source)
4. Add humidity and solar_radiation from NASA POWER (not in Meteostat)
5. Export merged dataset

## Dataset Comparison

### Before Merge (Meteostat Only)

| Feature | Missing Values | Percentage |
| ------- | -------------- | ---------- |
| tavg    | 1,047          | 26.8%      |
| tmin    | 1,080          | 27.6%      |
| tmax    | 1,047          | 26.8%      |
| prcp    | 1,063          | 27.2%      |
| wspd    | 1,109          | 28.3%      |
| wpgt    | 3,914          | 100.0%     |
| pres    | 1,115          | 28.5%      |

### After Merge

| Feature         | Missing Values | Percentage | Source           |
| --------------- | -------------- | ---------- | ---------------- |
| tavg            | 0              | 0%         | Meteostat + NASA |
| tmin            | 0              | 0%         | Meteostat + NASA |
| tmax            | 0              | 0%         | Meteostat + NASA |
| prcp            | 0              | 0%         | Meteostat + NASA |
| wspd            | 0              | 0%         | Meteostat + NASA |
| wpgt            | 0              | 0%         | Meteostat + NASA |
| pres            | 0              | 0%         | Meteostat + NASA |
| humidity        | 0              | 0%         | NASA only        |
| solar_radiation | 0              | 0%         | NASA only        |

## Advanced Usage

### Custom Date Ranges

Fetch specific date ranges for both sources:

```bash
# Fetch Meteostat data for 2020 only
python -m code.fetch_meteostat_weather --start-date 2020-01-01 --end-date 2020-12-31 --combine

# Fetch NASA POWER data for 2020 only
python -m code.fetch_nasa_power --start-date 2020-01-01 --end-date 2020-12-31 --combine

# Merge with custom paths
python -m code.merge_weather_data \
  --meteostat data/raw/weather_combined.csv \
  --nasa data/raw/nasa_power_combined.csv \
  --output data/processed/weather_merged_2020.csv
```

### Single Location

Fetch data for a single location:

```bash
python -m code.fetch_meteostat_weather --locations swat --combine
python -m code.fetch_nasa_power --locations swat --combine
python -m code.merge_weather_data
```

### Generate Synthetic NASA Data (for testing)

If NASA POWER API is not accessible, generate synthetic data:

```python
import pandas as pd
import numpy as np

# Load Meteostat data
meteo = pd.read_csv('data/raw/weather_combined.csv')

# Generate synthetic NASA data with similar patterns
# ... (see code/merge_weather_data.py comments for details)
```

## Validation

After merging, validate the dataset:

```python
import pandas as pd

df = pd.read_csv('data/processed/weather_merged.csv')

# Check for missing values
print("Missing values:", df.isnull().sum())

# Check data ranges
print("\nData statistics:")
print(df[['tavg', 'tmin', 'tmax', 'prcp', 'humidity']].describe())

# Verify all dates are covered
print("\nDate range:", df['date'].min(), "to", df['date'].max())
print("Total rows:", len(df))
```

## Next Steps

After merging weather data:

1. **Label flood events** from NDMA reports to create the target variable
2. **Feature engineering** - create rolling averages, cumulative rainfall, etc.
3. **Train ML models** using the complete dataset
4. **Deploy** the trained model for predictions

## Troubleshooting

### NASA API Connection Error

**Error:** `Failed to resolve 'power.larc.nasa.gov'`

**Solution:** The NASA POWER domain is blocked in your network. Options:
1. Use pre-generated NASA data in `data/raw/`
2. Run from a different network with API access
3. Contact your network administrator to whitelist `power.larc.nasa.gov`

### Date Mismatch

**Error:** Different date ranges in Meteostat vs NASA data

**Solution:** Ensure both datasets are fetched with the same `--start-date` and `--end-date` parameters.

### Missing Merged File

**Error:** `data/processed/weather_merged.csv not found`

**Solution:** The `data/processed/` directory is created automatically. Ensure you've run all three scripts in order:
1. `fetch_meteostat_weather`
2. `fetch_nasa_power`
3. `merge_weather_data`
