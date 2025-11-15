# AI-Based Natural Disaster Prediction Web App

An AI-powered web application for predicting natural disasters (specifically floods) in Pakistan using historical weather data and machine learning.

## Overview

This project collects and processes weather data from multiple sources to train machine learning models for flood prediction in high-risk districts of Pakistan (Swat and Upper Dir).

## Features

- **Multi-Source Weather Data Collection**
  - Meteostat API integration for historical weather data
  - NASA POWER API integration for satellite-derived meteorological data
  - Automatic data merging to fill missing values

- **Comprehensive Weather Features**
  - Temperature (average, min, max)
  - Precipitation and snowfall
  - Wind speed and gusts
  - Atmospheric pressure
  - Humidity (from NASA POWER)
  - Solar radiation (from NASA POWER)

- **Data Processing Pipeline**
  - Automated data fetching
  - Missing value imputation using NASA POWER data
  - Data validation and quality checks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zohaibkhan745/-AI-Based-Natural-Disaster-Prediction-Web-App-.git
cd -AI-Based-Natural-Disaster-Prediction-Web-App-
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete data pipeline:
```bash
python examples/complete_data_pipeline.py
```

### Step-by-Step Data Collection

1. **Fetch Meteostat Data:**
```bash
python -m code.fetch_meteostat_weather --combine
```

2. **Fetch NASA POWER Data:**
```bash
python -m code.fetch_nasa_power --combine
```

3. **Merge Datasets:**
```bash
python -m code.merge_weather_data
```

### Custom Date Ranges

Fetch data for specific time periods:
```bash
python -m code.fetch_meteostat_weather --start-date 2020-01-01 --end-date 2020-12-31 --combine
python -m code.fetch_nasa_power --start-date 2020-01-01 --end-date 2020-12-31 --combine
python -m code.merge_weather_data
```

### Single Location

Fetch data for a specific location:
```bash
python -m code.fetch_meteostat_weather --locations swat --combine
python -m code.fetch_nasa_power --locations swat --combine
```

## Data Sources

### 1. Meteostat (Primary Source)
- **API:** https://meteostat.net/
- **Coverage:** 2018-01-01 to present
- **Features:** Temperature, precipitation, wind, pressure, sunshine
- **Issue:** 26-28% missing values

### 2. NASA POWER (Secondary Source)
- **API:** https://power.larc.nasa.gov/
- **Coverage:** Complete coverage for requested dates
- **Features:** Temperature, precipitation, wind, pressure, humidity, solar radiation
- **Purpose:** Fill missing Meteostat values and add new features

### 3. NDMA Reports (Planned)
- **Source:** National Disaster Management Authority
- **Purpose:** Label flood events for supervised learning
- **Status:** Manual labeling required

## Project Structure

```
.
├── code/
│   ├── fetch_meteostat_weather.py  # Fetch Meteostat data
│   ├── fetch_nasa_power.py          # Fetch NASA POWER data
│   └── merge_weather_data.py        # Merge both datasets
├── data/
│   ├── raw/                          # Raw data from APIs
│   └── processed/                    # Processed and merged data
├── docs/
│   ├── data_report.md                # Data collection report
│   └── data_merge_guide.md          # Merging guide
├── examples/
│   └── complete_data_pipeline.py    # Full pipeline example
└── requirements.txt                  # Python dependencies
```

## Data Pipeline

```
┌─────────────────┐     ┌──────────────────┐
│ Meteostat API  │     │ NASA POWER API   │
│  (Primary)      │     │  (Secondary)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│  Merge & Fill Missing Values            │
│  - Use Meteostat as primary             │
│  - Fill gaps with NASA POWER            │
│  - Add humidity & solar radiation       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Complete Weather Dataset               │
│  - 0% missing values                    │
│  - 18 features                          │
│  - Ready for ML                         │
└─────────────────────────────────────────┘
```

## Data Quality

### Before Merge (Meteostat Only)
- 3,914 rows
- 26-28% missing values in key features
- No humidity or solar radiation data

### After Merge
- 3,914 rows
- **0% missing values** for all key features
- Added humidity and solar radiation columns
- Ready for machine learning

## Next Steps

1. ✅ Collect and merge weather data
2. ⏳ Label flood events from NDMA reports
3. ⏳ Engineer additional features (rolling averages, cumulative rainfall)
4. ⏳ Train machine learning models
5. ⏳ Build web application for predictions
6. ⏳ Deploy the application

## Documentation

- [Data Collection Report](docs/data_report.md) - Detailed information about data sources
- [Data Merge Guide](docs/data_merge_guide.md) - Step-by-step merging instructions

## Requirements

- Python 3.8+
- pandas
- numpy
- requests
- meteostat
- beautifulsoup4
- lxml
- geopy

See [requirements.txt](requirements.txt) for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of an AI-based natural disaster prediction system for Pakistan.

## Acknowledgments

- Meteostat for providing historical weather data
- NASA POWER for satellite-derived meteorological data
- NDMA for disaster event reports
