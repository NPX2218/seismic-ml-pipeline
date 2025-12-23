"""
catalog.py: Fetch and cache earthquake catalogs from USGS.

This module provides functions to fetch earthquake catalogs from USGS
based on station location and recording time from miniSEED files.

miniSEED file → Extract metadata → Get station coordinates → Query USGS → Save CSV
                     ↓                      ↓
              (station, time)          (cached locally)

Usage:
    python catalog.py <mseed_file>
    python catalog.py data/raw/example.mseed

"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

import requests
import pandas as pd
from obspy import read


STATION_CACHE_FILE = Path("data/cache/station_coords.json")


def load_station_cache() -> Dict[str, Dict[str, float]]:
    """Load cached station coordinates from file."""
    if STATION_CACHE_FILE.exists():
        with open(STATION_CACHE_FILE, 'r', encoding="utf-8") as file:
            result: Dict[str, Dict[str, float]] = json.load(file)
            return result
    return {}


def save_station_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """Save station coordinates to cache file."""
    STATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATION_CACHE_FILE, 'w', encoding="utf-8") as file:
        json.dump(cache, file, indent=2)


def get_station_coords(network: str, station: str) -> Dict[str, float]:
    """
    Get station coordinates from IRIS, with caching.

    Args:
        network: Network code (e.g., 'AK')
        station: Station code (e.g., 'MDM')

    Returns:
        Dictionary with 'lat' and 'lon' keys

    Raises:
        ValueError: If station not found
        requests.RequestException: If network request fails
    """
    cache_key = f"{network}.{station}"

    # Check cache first
    cache = load_station_cache()
    if cache_key in cache:
        print(f"(Using cached coords for {cache_key})")
        return cache[cache_key]

    # Query IRIS
    print(f"Fetching coords from IRIS for {cache_key}...")
    url = "https://service.iris.edu/fdsnws/station/1/query"
    params: Dict[str, Any] = {
        'network': network,
        'station': station,
        'format': 'text',
        'level': 'station',
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise ValueError(f"Station not found: {cache_key}")

    lines = response.text.strip().split('\n')
    if len(lines) < 2:
        raise ValueError(f"No data for: {cache_key}")

    parts = lines[1].split('|')
    coords: Dict[str, float] = {
        'lat': float(parts[2]),
        'lon': float(parts[3])
    }

    # Save to cache
    cache[cache_key] = coords
    save_station_cache(cache)

    return coords


# Earthquake Catalog (with caching)

def fetch_catalog(
    mseed_path: str,
    output_dir: str = "data/catalogs",
    radius_km: float = 500,
    min_magnitude: float = 2.0,
    buffer_hours: float = 48,
    force_refresh: bool = False
) -> str:
    """
    Fetch USGS earthquake catalog for a given miniSEED recording.

    Args:
        mseed_path: Path to the miniSEED file
        output_dir: Directory to save the catalog CSV
        radius_km: Search radius around station in kilometers
        min_magnitude: Minimum earthquake magnitude to include
        buffer_hours: Hours before/after recording to include
        force_refresh: If True, fetch fresh data even if cached

    Returns:
        Path to the saved catalog CSV file

    Raises:
        RuntimeError: If USGS query fails
        ValueError: If station coordinates cannot be found
        requests.RequestException: If network request fails
    """
    # Read mseed metadata
    st = read(mseed_path, headonly=True)

    station = st[0].stats.station
    network = st[0].stats.network
    start_time = min(tr.stats.starttime for tr in st)
    end_time = max(tr.stats.endtime for tr in st)

    print(f"Station: {network}.{station}")
    print(f"Recording: {start_time} to {end_time}")

    # Check if catalog already exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    recording_date = start_time.strftime("%Y%m%d")
    output_file = Path(output_dir) / f"catalog_{station}_{recording_date}.csv"

    if output_file.exists() and not force_refresh:
        print(f"\n(Using cached catalog: {output_file})")
        df = pd.read_csv(output_file)
        print(f"Found {len(df)} earthquakes (cached)")
        return str(output_file)

    # Get coordinates (cached)
    coords = get_station_coords(network, station)
    print(f"Location: {coords['lat']:.2f}N, {coords['lon']:.2f}E")

    # Query USGS
    query_start = start_time - (buffer_hours * 3600)
    query_end = end_time + (buffer_hours * 3600)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params: Dict[str, Any] = {
        'format': 'csv',
        'starttime': query_start.isoformat(),
        'endtime': query_end.isoformat(),
        'latitude': coords['lat'],
        'longitude': coords['lon'],
        'maxradiuskm': radius_km,
        'minmagnitude': min_magnitude,
        'orderby': 'time',
    }

    print("\nQuerying USGS...")
    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"USGS query failed: {response.status_code}")

    # Save catalog
    with open(output_file, 'w', encoding="utf-8") as file:
        file.write(response.text)

    df = pd.read_csv(output_file)
    print(f"Found {len(df)} earthquakes")
    print(f"Saved to: {output_file}")

    return str(output_file)


def main() -> None:
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python catalog.py <mseed_file>")
        print("Example: python catalog.py data/raw/example.mseed")
        sys.exit(1)

    fetch_catalog(sys.argv[1])


if __name__ == "__main__":
    main()
