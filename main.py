"""
main.py — Run the feature extraction pipeline for seismic data analysis.

This module provides the main entry point for extracting spectral features
from seismic data using a sliding window approach. It supports both real
seismic data files (miniSEED format) and synthetic test data generation.

Usage:
    python main.py

Configuration:
    Set USE_TEST_DATA = True to test with synthetic data (no file needed).
    Modify INPUT_FILE, OUTPUT_FILE, TD, DELTA_T, and CHANNELS as needed.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from src.loader.loader import load_file

from src.helpers.helper import print_divider


def generate_test_data(
    n_channels: int = 3,
    duration_hours: float = 10,
    sampling_rate: int = 50
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """
    Generate synthetic seismic data for testing the feature extraction pipeline.

    Creates multi-channel synthetic seismic signals composed of multiple
    frequency components (low, mid, high) with added Gaussian noise.

    Args:
        n_channels: Number of seismic channels to generate. Each channel
            receives a slightly phase-shifted version of the signal.
        duration_hours: Total duration of the synthetic data in hours.
        sampling_rate: Sampling frequency in Hz (samples per second).
    """
    pass


# Set to True to test with synthetic data (no file needed)
USE_TEST_DATA: bool = False

INPUT_FILE: str = "data/raw/example.mseed"
OUTPUT_FILE: str = "features.npy"

TD: float = 2.0        # Window duration in hours
DELTA_T: float = 0.5   # Step size in hours

# Channel selection (None = all channels)
CHANNELS: list[int] | None = None


def main() -> None:
    """
    Execute the complete feature extraction pipeline.

    This function orchestrates the entire feature extraction workflow:

    1. Data Loading: Either generates synthetic test data or loads
       real seismic data from a miniSEED file.

    2. Window Calculation: Computes the number of sliding windows
       based on data duration and window parameters.

    3. Feature Extraction: Iterates through each window and extracts
       spectral features (centroid and spread) for multiple frequency bands.

    4. Output: Saves the feature matrix to a NumPy file and feature
       names to a text file.

    Returns:
        None. Results are saved to disk.

    Raises:
        Prints error messages and returns early if:
            - Input file not found (when USE_TEST_DATA is False)
            - File loading fails
            - Insufficient data for windowing
            - No features extracted

    Output Files:
        - {OUTPUT_FILE}: NumPy array of shape (n_windows, n_features)
        - {OUTPUT_FILE}_names.txt: Feature names, one per line

    """

    print_divider()
    print("STEP 1: Loading data")
    print_divider()

    data: NDArray[np.float64]
    metadata: dict[str, Any]

    if USE_TEST_DATA:
        # Generate synthetic data for testing
        data, metadata = generate_test_data(
            n_channels=3,
            duration_hours=10,
            sampling_rate=50
        )
    else:
        # Load from file
        if not Path(INPUT_FILE).exists():
            print("ERROR: File not found: {INPUT_FILE}")
            print("Set USE_TEST_DATA = True to test with synthetic data")
            return

        try:
            data, metadata = load_file(
                INPUT_FILE, channels=CHANNELS, max_length_diff=100)
        except FileNotFoundError:
            print(f"  ERROR: File not found: {INPUT_FILE}")
            return
        except ValueError as e:
            print(f"  ERROR: Invalid data: {e}")
            return

        sampling_rate = int(metadata['sampling_rate'])
        n_channels = int(metadata['n_channels'])
        n_samples = int(metadata['n_samples'])
        gaps = cast(list[tuple[float, float]], metadata['gaps'])

        duration_hours: float = n_samples / sampling_rate / 3600

        print(f"File: {INPUT_FILE}")
        print(f"Data shape: {data.shape}")
        print(f"Channels: {n_channels}")
        print(f"Samples: {n_samples:,}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Duration: {duration_hours:.2f} hours")
        print(f"Gaps: {len(gaps)} (will skip windows containing these)")
        print()


if __name__ == "__main__":
    main()
