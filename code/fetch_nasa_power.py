"""Download daily NASA POWER meteorological data for key Pakistani districts.

Example usage (run from repo root):

    python -m code.fetch_nasa_power --combine
    python -m code.fetch_nasa_power --locations swat --start-date 2021-01-01 --end-date 2021-12-31

The script mirrors the location configuration used by ``fetch_meteostat_weather`` and saves
per-location CSVs (plus an optional combined file) under ``data/raw``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

DATE_FMT = "%Y-%m-%d"
NASA_DATE_FMT = "%Y%m%d"
DEFAULT_START = datetime(2018, 1, 1)
DEFAULT_END = datetime.now()
OUTPUT_DIR = Path("data/raw")
COMBINED_FILENAME = "nasa_power_combined.csv"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "PRECTOTCORR",
    "WS2M",
    "WS2M_MAX",
    "RH2M",
    "PS",
    "ALLSKY_SFC_SW_DWN",
]


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int


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

COLUMN_MAP = {
    "T2M": "nasa_tavg",
    "T2M_MIN": "nasa_tmin",
    "T2M_MAX": "nasa_tmax",
    "PRECTOTCORR": "nasa_prcp",
    "WS2M": "nasa_wspd",
    "WS2M_MAX": "nasa_wpgt",
    "RH2M": "humidity",
    "PS": "nasa_pres",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
}


class NasaPowerError(RuntimeError):
    pass


def _parse_date(value: str, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, DATE_FMT)
    except ValueError as exc:
        raise ValueError(f"{label} must follow YYYY-MM-DD format, got: {value}") from exc
    return parsed


def _request_payload(location: Location, start: datetime, end: datetime) -> Dict[str, str]:
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "ag",
        "longitude": f"{location.longitude}",
        "latitude": f"{location.latitude}",
        "start": start.strftime(NASA_DATE_FMT),
        "end": end.strftime(NASA_DATE_FMT),
        "format": "JSON",
    }


def fetch_nasa_power(location: Location, start: datetime, end: datetime) -> pd.DataFrame:
    params = _request_payload(location, start, end)
    logging.info(
        "Fetching NASA POWER for %s (%s–%s)", location.name, start.strftime(DATE_FMT), end.strftime(DATE_FMT)
    )
    response = requests.get(BASE_URL, params=params, timeout=60)
    if response.status_code != 200:
        raise NasaPowerError(f"NASA POWER API error {response.status_code}: {response.text[:200]}")
    payload = response.json()
    if "properties" not in payload or "parameter" not in payload["properties"]:
        raise NasaPowerError("Unexpected NASA POWER response structure")
    parameters = payload["properties"]["parameter"]
    if not parameters:
        raise NasaPowerError("NASA POWER response missing parameter data")

    all_dates = set()
    for param_values in parameters.values():
        all_dates.update(param_values.keys())
    if not all_dates:
        raise NasaPowerError("NASA POWER returned no daily rows for the requested range")

    rows = []
    for date_str in sorted(all_dates):
        row: Dict[str, float] = {}
        for param, values in parameters.items():
            value = values.get(date_str)
            column = COLUMN_MAP.get(param, param.lower())
            row[column] = value
        row["date"] = datetime.strptime(date_str, NASA_DATE_FMT).strftime(DATE_FMT)
        row["location_key"] = location.key
        row["location_name"] = location.name
        row["latitude"] = location.latitude
        row["longitude"] = location.longitude
        row["elevation_m"] = location.elevation_m
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date")
    return df


def export_location_csv(df: pd.DataFrame, location: Location, output_dir: Path, start: datetime, end: datetime) -> Path:
    filename = f"nasa_power_{location.key}_{start.date()}_{end.date()}.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows)", path, len(df))
    return path


def export_combined_csv(frames: List[pd.DataFrame], output_dir: Path) -> Path:
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "location_key"])
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
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(LOCATIONS.keys()),
        help="Location keys to download (default: swat upper_dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for CSV exports",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Write data/raw/nasa_power_combined.csv with all rows",
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

    start = _parse_date(args.start_date, "--start-date")
    end = _parse_date(args.end_date, "--end-date")
    if start > end:
        raise ValueError("--start-date must be earlier than or equal to --end-date")

    requested = [key.lower() for key in args.locations]
    invalid = [key for key in requested if key not in LOCATIONS]
    if invalid:
        raise KeyError(f"Unknown location key(s): {invalid}. Valid: {list(LOCATIONS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for key in requested:
        location = LOCATIONS[key]
        df = fetch_nasa_power(location, start, end)
        export_location_csv(df, location, output_dir, start, end)
        frames.append(df)

    if args.combine and frames:
        export_combined_csv(frames, output_dir)

    logging.info("Finished downloading NASA POWER data for %d location(s)", len(frames))


if __name__ == "__main__":
    main()
