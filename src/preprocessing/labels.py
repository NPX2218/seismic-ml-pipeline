"""
labels.py: Generate labels from earthquake catalog
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from src.preprocessing.windows import Window


def load_catalog(filepath: str, data_start_time: str, min_magnitude: float = 0.0) -> pd.DataFrame:
    """
    Load earthquake catalog and convert times to hours from data start.

    Args:
        filepath: Path to USGS CSV file
        data_start_time: Start time of your seismic data (UTC string)
        min_magnitude: Only include earthquakes >= this magnitude

    Returns:
        DataFrame with 'hours_from_start' and 'mag' columns
    """
    catalog = pd.read_csv(filepath)

    # Filter by magnitude
    catalog = catalog[catalog['mag'] >= min_magnitude].copy()

    # Parse data start time
    data_start = UTCDateTime(data_start_time)

    # Convert earthquake times to hours from data start
    hours_from_start = []
    for eq_time in catalog['time']:
        eq_utc = UTCDateTime(eq_time)
        hours = (eq_utc - data_start) / 3600
        hours_from_start.append(hours)

    catalog['hours_from_start'] = hours_from_start

    return catalog


def generate_labels(windows: List[Window],
                    catalog: pd.DataFrame,
                    lookahead_hours: float = 24.0) -> np.ndarray:
    """
    Label windows based on whether an earthquake is coming.

    Args:
        windows: List of Window objects
        catalog: DataFrame with 'hours_from_start' column
        lookahead_hours: Label is 1 if earthquake within this time

    Returns:
        labels: Array of 0s and 1s
    """
    eq_times = catalog['hours_from_start'].values
    labels = []

    for w in windows:
        label = 0
        for eq_time in eq_times:
            # Earthquake is within lookahead window after this window ends?
            if w.end_hours < eq_time <= w.end_hours + lookahead_hours:
                label = 1
                break
        labels.append(label)

    return np.array(labels)


def get_label_stats(labels: np.ndarray) -> Dict[str, Any]:
    """Get statistics about the labels."""
    return {
        'total': len(labels),
        'positive': int(np.sum(labels)),
        'negative': int(len(labels) - np.sum(labels)),
        'positive_pct': float(np.sum(labels) / len(labels) * 100)
    }
