"""
Clean Missing Values in Flood Weather Dataset
==============================================
This script handles missing values appropriately for ML model training.
"""

import pandas as pd
import numpy as np

def clean_dataset(input_path, output_path):
    """
    Clean the flood weather dataset by handling missing values.
    
    Strategy:
    1. DROP columns with >95% missing (useless for ML)
    2. DROP metadata columns not needed for prediction
    3. IMPUTE NASA columns using Meteostat equivalents
    """
    
    print("=" * 60)
    print("CLEANING FLOOD WEATHER DATASET")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(input_path, low_memory=False)
    print(f"\nOriginal dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Original missing values: {df.isnull().sum().sum():,}")
    
    # =========================================================================
    # STEP 1: DROP columns with >95% missing (completely useless)
    # =========================================================================
    columns_to_drop = [
        'tsun',              # 100% missing - sunshine duration (no data)
        'snow',              # 99.97% missing - irrelevant for Pakistan
        'damages_inr_crore', # 99.97% missing - only for flood events
        'warnings',          # 99.97% missing - metadata
        'source_url',        # 99.97% missing - metadata
        'notes',             # 99.97% missing - text metadata
        'flood_severity',    # 97.26% missing - only for flood events
        'flood_source',      # 97.26% missing - metadata
        'flood_notes',       # 97.26% missing - metadata
        'location_id',       # 69.57% missing - redundant (have location_key)
    ]
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    print(f"\n[STEP 1] Dropping {len(columns_to_drop)} columns with >95% missing or metadata:")
    for col in columns_to_drop:
        missing_pct = (df[col].isnull().sum() / len(df) * 100)
        print(f"  - {col}: {missing_pct:.2f}% missing")
    
    df = df.drop(columns=columns_to_drop)
    
    # =========================================================================
    # STEP 2: IMPUTE NASA columns using Meteostat equivalents
    # =========================================================================
    print(f"\n[STEP 2] Imputing NASA columns with Meteostat equivalents:")
    
    # Mapping: NASA column -> Meteostat equivalent
    nasa_to_meteostat = {
        'nasa_tavg': 'tavg',
        'nasa_tmin': 'tmin',
        'nasa_tmax': 'tmax',
        'nasa_prcp': 'prcp',
        'nasa_wspd': 'wspd',
        'nasa_wpgt': 'wpgt',
        'nasa_pres': 'pres',
    }
    
    for nasa_col, meteo_col in nasa_to_meteostat.items():
        if nasa_col in df.columns and meteo_col in df.columns:
            missing_before = df[nasa_col].isnull().sum()
            # Fill NASA missing values with Meteostat values
            df[nasa_col] = df[nasa_col].fillna(df[meteo_col])
            missing_after = df[nasa_col].isnull().sum()
            print(f"  - {nasa_col}: {missing_before:,} -> {missing_after:,} missing (filled from {meteo_col})")
    
    # =========================================================================
    # STEP 3: Handle any remaining missing values
    # =========================================================================
    print(f"\n[STEP 3] Checking for remaining missing values:")
    
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    
    if len(remaining_missing) > 0:
        print("  Remaining columns with missing values:")
        for col, count in remaining_missing.items():
            pct = count / len(df) * 100
            print(f"    - {col}: {count:,} ({pct:.2f}%)")
            
            # Fill numeric columns with median
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
                print(f"      -> Filled with median: {df[col].median():.2f}")
    else:
        print("  No remaining missing values!")
    
    # =========================================================================
    # STEP 4: Final verification
    # =========================================================================
    print(f"\n[STEP 4] Final dataset statistics:")
    print(f"  - Rows: {df.shape[0]:,}")
    print(f"  - Columns: {df.shape[1]}")
    print(f"  - Total missing values: {df.isnull().sum().sum()}")
    print(f"  - Flood events: {df['flood_event'].sum():,} ({df['flood_event'].mean()*100:.2f}%)")
    
    # List final columns
    print(f"\n[FINAL COLUMNS] ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"  {i:2}. {col} ({dtype})")
    
    # =========================================================================
    # SAVE cleaned dataset
    # =========================================================================
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Cleaned dataset saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)
    
    return df


def main():
    input_path = r'd:\SEMESTER-5\CS351(Artificial Intelligence)\AI-Based Natural Disaster\data\processed\flood_weather_dataset.csv'
    output_path = r'd:\SEMESTER-5\CS351(Artificial Intelligence)\AI-Based Natural Disaster\data\processed\flood_weather_dataset_cleaned.csv'
    
    df = clean_dataset(input_path, output_path)
    
    # Show comparison
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    
    original = pd.read_csv(input_path, low_memory=False)
    print(f"\nBEFORE: {original.shape[1]} columns, {original.isnull().sum().sum():,} missing values")
    print(f"AFTER:  {df.shape[1]} columns, {df.isnull().sum().sum()} missing values")
    print(f"\nRemoved {original.shape[1] - df.shape[1]} useless columns")
    print(f"Eliminated {original.isnull().sum().sum():,} missing values")


if __name__ == "__main__":
    main()
