"""Merge Meteostat and NASA POWER data to create a complete weather dataset.

This script combines weather data from two sources:
1. Meteostat (primary source) - provides detailed weather measurements but with gaps
2. NASA POWER (secondary source) - fills missing values with satellite-derived data

Usage examples (run from repo root):

    python -m code.merge_weather_data
    python -m code.merge_weather_data --meteostat data/raw/weather_combined.csv --nasa data/raw/nasa_power_combined.csv
    python -m code.merge_weather_data --output data/processed/weather_merged.csv

The script will:
- Load both Meteostat and NASA POWER datasets
- Align them by date and location
- Fill missing Meteostat values with NASA POWER data
- Add humidity and solar_radiation columns from NASA POWER
- Export the merged dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_METEOSTAT = "data/raw/weather_combined.csv"
DEFAULT_NASA = "data/raw/nasa_power_combined.csv"
DEFAULT_OUTPUT = "data/processed/weather_merged.csv"

# Mapping between Meteostat columns and NASA POWER columns
MERGE_MAPPING = {
    "tavg": "nasa_tavg",
    "tmin": "nasa_tmin",
    "tmax": "nasa_tmax",
    "prcp": "nasa_prcp",
    "wspd": "nasa_wspd",
    "wpgt": "nasa_wpgt",
    "pres": "nasa_pres",
}


def load_meteostat(path: Path) -> pd.DataFrame:
    """Load Meteostat weather data."""
    logging.info("Loading Meteostat data from %s", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    logging.info("Loaded %d rows from Meteostat", len(df))
    return df


def load_nasa_power(path: Path) -> pd.DataFrame:
    """Load NASA POWER weather data."""
    logging.info("Loading NASA POWER data from %s", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    logging.info("Loaded %d rows from NASA POWER", len(df))
    return df


def merge_weather_data(meteostat_df: pd.DataFrame, nasa_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Meteostat and NASA POWER data, filling missing values.
    
    Args:
        meteostat_df: DataFrame from Meteostat with weather data
        nasa_df: DataFrame from NASA POWER with weather data
        
    Returns:
        Merged DataFrame with filled missing values
    """
    logging.info("Starting merge process...")
    
    # Create a copy to avoid modifying the original
    merged = meteostat_df.copy()
    
    # Merge on date and location_key
    nasa_subset = nasa_df[["date", "location_key"] + list(MERGE_MAPPING.values()) + ["humidity", "solar_radiation"]]
    merged = merged.merge(
        nasa_subset,
        on=["date", "location_key"],
        how="left",
        suffixes=("", "_nasa_temp")
    )
    
    # Count missing values before merge
    missing_before = {}
    for meteo_col in MERGE_MAPPING.keys():
        missing_before[meteo_col] = merged[meteo_col].isnull().sum()
    
    # Fill missing values using NASA POWER data
    for meteo_col, nasa_col in MERGE_MAPPING.items():
        if meteo_col in merged.columns and nasa_col in merged.columns:
            # Fill missing Meteostat values with NASA POWER values
            mask = merged[meteo_col].isnull() & merged[nasa_col].notnull()
            merged.loc[mask, meteo_col] = merged.loc[mask, nasa_col]
            
            filled = mask.sum()
            if filled > 0:
                logging.info("Filled %d missing values in %s using %s", filled, meteo_col, nasa_col)
    
    # Count missing values after merge
    missing_after = {}
    for meteo_col in MERGE_MAPPING.keys():
        missing_after[meteo_col] = merged[meteo_col].isnull().sum()
    
    # Log summary statistics
    logging.info("\n=== Missing Values Summary ===")
    for meteo_col in MERGE_MAPPING.keys():
        before = missing_before[meteo_col]
        after = missing_after[meteo_col]
        filled = before - after
        if before > 0:
            pct_filled = (filled / before) * 100
            logging.info(
                "%s: %d → %d (filled %d, %.1f%%)",
                meteo_col, before, after, filled, pct_filled
            )
    
    # Drop NASA POWER temporary columns (keep only humidity and solar_radiation)
    nasa_cols_to_drop = [col for col in MERGE_MAPPING.values() if col in merged.columns]
    merged = merged.drop(columns=nasa_cols_to_drop)
    
    # Reorder columns for better readability
    base_cols = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]
    meta_cols = ["location_key", "location_name", "latitude", "longitude", "elevation_m"]
    nasa_cols = ["humidity", "solar_radiation"]
    
    # Only include columns that exist
    ordered_cols = [col for col in base_cols + nasa_cols + meta_cols if col in merged.columns]
    other_cols = [col for col in merged.columns if col not in ordered_cols]
    merged = merged[ordered_cols + other_cols]
    
    logging.info("Merge complete. Final dataset has %d rows and %d columns", len(merged), len(merged.columns))
    return merged


def export_merged_data(df: pd.DataFrame, output_path: Path) -> None:
    """Export merged weather data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Exported merged data to %s (%d rows)", output_path, len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--meteostat",
        default=DEFAULT_METEOSTAT,
        help=f"Path to Meteostat combined CSV (default: {DEFAULT_METEOSTAT})",
    )
    parser.add_argument(
        "--nasa",
        default=DEFAULT_NASA,
        help=f"Path to NASA POWER combined CSV (default: {DEFAULT_NASA})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for merged CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    
    meteostat_path = Path(args.meteostat)
    nasa_path = Path(args.nasa)
    output_path = Path(args.output)
    
    # Validate input files exist
    if not meteostat_path.exists():
        raise FileNotFoundError(f"Meteostat file not found: {meteostat_path}")
    if not nasa_path.exists():
        raise FileNotFoundError(f"NASA POWER file not found: {nasa_path}")
    
    # Load data
    meteostat_df = load_meteostat(meteostat_path)
    nasa_df = load_nasa_power(nasa_path)
    
    # Merge data
    merged_df = merge_weather_data(meteostat_df, nasa_df)
    
    # Export merged data
    export_merged_data(merged_df, output_path)
    
    logging.info("Weather data merge completed successfully!")


if __name__ == "__main__":
    main()
