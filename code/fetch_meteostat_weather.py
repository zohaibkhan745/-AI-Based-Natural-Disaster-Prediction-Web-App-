"""Utility to download daily weather features for key Pakistani districts via Meteostat.

Usage examples (run from repo root):

    python -m code.fetch_meteostat_weather --start-date 2018-01-01 --end-date 2023-12-31
    python -m code.fetch_meteostat_weather --locations swat --combine

The script fetches data for Swat and Upper Dir by default and stores individual CSVs
inside ``data/raw``. Pass ``--combine`` to emit an additional merged file containing
all selected locations in a single table.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from meteostat import Daily, Point

DATE_FMT = "%Y-%m-%d"
DEFAULT_START = datetime(2018, 1, 1)
DEFAULT_END = datetime.now()
OUTPUT_DIR = Path("data/raw")
COMBINED_FILENAME = "weather_combined.csv"


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int

    def to_point(self) -> Point:
        return Point(self.latitude, self.longitude, self.elevation_m)


LOCATIONS: Dict[str, Location] = {
    "swat": Location(
        key="swat",
        name="Swat District, Khyber Pakhtunkhwa",
        latitude=34.8091,
        longitude=72.3617,
        elevation_m=980,
    ),
    "upper_dir": Location(
        key="upper_dir",
        name="Upper Dir District, Khyber Pakhtunkhwa",
        latitude=35.3350,
        longitude=71.8760,
        elevation_m=1420,
    ),
}


def _parse_date(value: str, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, DATE_FMT)
    except ValueError as exc:
        raise ValueError(f"{label} must follow YYYY-MM-DD format, got: {value}") from exc
    return parsed


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_daily_weather(location: Location, start: datetime, end: datetime) -> pd.DataFrame:
    logging.info("Fetching %s (%s–%s)", location.name, start.date(), end.date())
    daily = Daily(location.to_point(), start, end)
    df = daily.fetch()

    if df.empty:
        raise RuntimeError(
            f"Meteostat returned no rows for {location.name} in range {start.date()}–{end.date()}"
        )

    df = df.reset_index().rename(columns={"time": "date"})
    df["location_key"] = location.key
    df["location_name"] = location.name
    df["latitude"] = location.latitude
    df["longitude"] = location.longitude
    df["elevation_m"] = location.elevation_m

    return df


def export_location_csv(df: pd.DataFrame, location: Location, output_dir: Path, start: datetime, end: datetime) -> Path:
    filename = f"weather_{location.key}_{start.date()}_{end.date()}.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows)", path, len(df))
    return path


def export_combined_csv(frames: List[pd.DataFrame], output_dir: Path) -> Path:
    combined = pd.concat(frames, ignore_index=True).sort_values("date")
    path = output_dir / COMBINED_FILENAME
    combined.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows)", path, len(combined))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START.strftime(DATE_FMT),
        help=f"Start date in {DATE_FMT} format (default: {DEFAULT_START.strftime(DATE_FMT)})",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END.strftime(DATE_FMT),
        help=f"End date in {DATE_FMT} format (default: today)",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(LOCATIONS.keys()),
        help="One or more location keys to download (default: swat upper_dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for the exported CSV files",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all selected locations into data/raw/weather_combined.csv",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    start = _parse_date(args.start_date, "--start-date")
    end = _parse_date(args.end_date, "--end-date")
    if start > end:
        raise ValueError("--start-date must be earlier than or equal to --end-date")

    requested = [key.lower() for key in args.locations]
    invalid = [key for key in requested if key not in LOCATIONS]
    if invalid:
        raise KeyError(f"Unknown location key(s): {invalid}. Valid keys: {list(LOCATIONS)}")

    output_dir = _ensure_output_dir(Path(args.output_dir))

    exported_frames: List[pd.DataFrame] = []
    for key in requested:
        location = LOCATIONS[key]
        frame = fetch_daily_weather(location, start, end)
        export_location_csv(frame, location, output_dir, start, end)
        exported_frames.append(frame)

    if args.combine and exported_frames:
        export_combined_csv(exported_frames, output_dir)

    logging.info("Finished downloading weather data for %d location(s)", len(exported_frames))


if __name__ == "__main__":
    main()
