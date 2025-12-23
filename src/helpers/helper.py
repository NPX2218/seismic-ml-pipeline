"""
helper.py: Contains functions that can help debugging or other helpful functions.
"""
from typing import Dict, Any
import json

from obspy import read
import requests


def print_divider() -> None:
    """Print a visual divider line for console output formatting."""
    print("=" * 60)


def inspect_file(filepath: str) -> None:
    """
    Print detailed info about each trace in a file without loading all data.

    Useful for exploring a file's structure before deciding which 
    channels to load or to debug issues with fragmented traces.

    Args:
        filepath: Path to the .mseed file to inspect.
    """
    st = read(filepath, headonly=True)

    print(f"File: {filepath}")
    print(f"Total traces: {len(st)}")
    print("-" * 70)

    for i, trace in enumerate(st):
        stats = trace.stats
        print(stats)
        print(f"Trace {i}:")
        print(f"Station: {stats.station}")
        print(f"  Channel: {stats.channel}")
        print(f"  Start: {stats.starttime}")
        print(f"  End: {stats.endtime}")
        print(f"  Samples: {stats.npts:,}")
        print()


def download_all_stations(network: str, output_file: str | None = None) -> Dict[Any, Any]:
    """
    Download all station coordinates for a network.

    Args:
        network: Network code (e.g., 'AK' for Alaska)

    Returns:
        Dict of {station_name: {'lat': lat, 'lon': lon}}
    """

    url = "https://service.iris.edu/fdsnws/station/1/query"

    params = {
        'network': network,
        'format': 'text',
        'level': 'station',
    }

    response = requests.get(url, params=params, timeout=10)

    stations = {}
    lines = response.text.strip().split('\n')

    for line in lines[1:]:  # Skip header
        parts = line.split('|')
        if len(parts) >= 4:
            station_name = parts[1]
            lat = float(parts[2])
            lon = float(parts[3])
            stations[station_name] = {'lat': lat, 'lon': lon}

    # Optionally save to file
    if output_file:
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(stations, f, indent=2)
        print(f"Saved {len(stations)} stations to {output_file}")

    return stations


"""
stations = download_all_stations('AV', 'data/stations/AV_stations.json')
print(f"Found {len(stations)} stations")

stations = download_all_stations('AK', 'data/stations/AK_stations.json')
"""
