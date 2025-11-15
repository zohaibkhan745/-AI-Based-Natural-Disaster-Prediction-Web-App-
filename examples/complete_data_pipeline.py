"""Complete data pipeline example for weather data collection and merging.

This script demonstrates the full workflow:
1. Fetch Meteostat weather data
2. Fetch NASA POWER data
3. Merge both datasets to fill missing values
4. Generate summary statistics

Usage:
    python examples/complete_data_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and display its output."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: Command failed with exit code {result.returncode}")
        return result.returncode
    
    print(f"✅ {description} completed successfully")
    return 0


def display_statistics(csv_path: Path, title: str) -> None:
    """Display summary statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")
    
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({missing_pct[col]}%)")
    
    if missing.sum() == 0:
        print("  ✅ No missing values!")
    
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Locations: {', '.join(df['location_key'].unique())}")


def main() -> int:
    """Run the complete data pipeline."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Weather Data Collection and Merging Pipeline               ║
║  AI-Based Natural Disaster Prediction Web App                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Fetch Meteostat data
    ret = run_command(
        [sys.executable, "-m", "code.fetch_meteostat_weather", "--combine"],
        "Fetch Meteostat weather data"
    )
    if ret != 0:
        return ret
    
    display_statistics(
        Path("data/raw/weather_combined.csv"),
        "Meteostat Data Summary"
    )
    
    # Step 2: Fetch NASA POWER data
    print("\n⚠️  NOTE: NASA POWER API may be blocked in some environments.")
    print("If the next step fails, pre-generated NASA data will be used.\n")
    
    ret = run_command(
        [sys.executable, "-m", "code.fetch_nasa_power", "--combine"],
        "Fetch NASA POWER data"
    )
    # Don't fail if NASA fetch fails - we might have pre-generated data
    if ret != 0:
        print("⚠️  NASA POWER fetch failed, checking for existing data...")
        if not Path("data/raw/nasa_power_combined.csv").exists():
            print("❌ No NASA POWER data available. Please generate synthetic data first.")
            return 1
        print("✅ Using existing NASA POWER data")
    
    display_statistics(
        Path("data/raw/nasa_power_combined.csv"),
        "NASA POWER Data Summary"
    )
    
    # Step 3: Merge datasets
    ret = run_command(
        [sys.executable, "-m", "code.merge_weather_data"],
        "Merge Meteostat and NASA POWER data"
    )
    if ret != 0:
        return ret
    
    display_statistics(
        Path("data/processed/weather_merged.csv"),
        "Merged Data Summary"
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("Pipeline completed successfully! 🎉")
    print(f"{'='*60}\n")
    print("Next steps:")
    print("1. Label flood events from NDMA reports")
    print("2. Engineer additional features (rolling averages, etc.)")
    print("3. Train machine learning models")
    print("\nMerged dataset location:")
    print("  data/processed/weather_merged.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
