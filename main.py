"""
main.py: Run the feature extraction pipeline for seismic data analysis.

This module provides the main entry point for extracting spectral features
from seismic data using a sliding window approach.

Usage:
    python main.py                     # Process all .mseed files in data/raw/
    python main.py data/raw/file.mseed # Process a single file
    python main.py /path/to/folder     # Process all .mseed files in folder

Configuration:
    Modify TD, DELTA_T, and CHANNELS as needed.
"""

import argparse
from pathlib import Path
from typing import Any, cast
import json
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from obspy import read
from pandas import DataFrame
import requests

from src.loader.loader import load_file
from src.preprocessing.windows import window_generator, count_windows
from src.preprocessing.features import compute_bin_edges, extract_features_batch
from src.preprocessing.labels import load_catalog, generate_labels, get_label_stats

from src.preprocessing.catalog import fetch_catalog
from src.helpers.helper import print_divider


# Pipeline Configuration

TD: float = 2.0              # Window duration in hours
DELTA_T: float = 0.5         # Step size in hours
N_BINS: int = 100            # Number of histogram bins
LOOKAHEAD_HOURS: float = 6.0  # Label lookahead window
MIN_MAGNITUDE: float = 3.0   # Minimum earthquake magnitude for labels

# Channel selection (None = all channels)
CHANNELS: list[int] | None = None


def process_file(input_file: str) -> bool:
    """
    Process a single mseed file through the feature extraction pipeline.

    This function orchestrates the entire feature extraction workflow:

    1. Data Loading: Loads seismic data from a miniSEED file.
    2. Window Calculation: Computes the number of sliding windows
       based on data duration and window parameters.
    3. Bin Edges: Computes bin edges from the full signal for consistent
       amplitude scaling across all windows.
    4. Feature Extraction: Extracts spectral features (centroid and spread)
       for multiple frequency bands from each window.
    5. Output: Saves the feature matrix, timestamps, and config to disk.
    6. Labels: Generates and saves labels for machine learning training.

    Args:
        input_file: Path to the miniSEED file to process.

    Returns:
        True if processing succeeded, False otherwise.
    """

    print_divider()
    print("STEP 1: Loading data")
    print_divider()

    data: NDArray[np.float64]
    metadata: dict[str, Any]

    # Load from file
    if not Path(input_file).exists():
        print(f"ERROR: File not found: {input_file}")
        return False

    try:
        data, metadata = load_file(
            input_file, channels=CHANNELS, max_length_diff=100)
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        return False
    except ValueError as e:
        print(f"ERROR: Invalid data: {e}")
        return False

    sampling_rate = int(metadata['sampling_rate'])
    n_channels = int(metadata['n_channels'])
    n_samples = int(metadata['n_samples'])
    gaps = cast(list[tuple[float, float]], metadata['gaps'])

    duration_hours: float = n_samples / sampling_rate / 3600

    print(f"File: {input_file}")
    print(f"Data shape: {data.shape}")
    print(f"Channels: {n_channels}")
    print(f"Samples: {n_samples:,}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Gaps: {len(gaps)} (will skip windows containing these)")
    print()

    # STEP 2: Window Calculation

    print("=" * 60)
    print("STEP 2: Window calculation")
    print("=" * 60)

    n_windows: int = count_windows(n_samples, TD, DELTA_T, sampling_rate)

    print(f"Window duration: {TD} hours")
    print(f"Step size: {DELTA_T} hours")
    print(f"Number of windows: {n_windows}")

    if n_windows == 0:
        print()
        print("ERROR: Not enough data for even one window!")
        print(f"Data duration: {duration_hours:.2f} hours")
        print(f"Window size: {TD} hours")
        print(f"Need at least {TD} hours of data.")
        return False

    # channels × bands × (centroid, spread)
    n_features: int = n_channels * 3 * 2
    print(f"Features per window: {n_features}")
    print()

    # STEP 3: Computing bin edge size

    print_divider()
    print("STEP 3: Computing bin edges from full signal")
    print_divider()

    # Bin edges must be computed ONCE from entire signal
    # This ensures all windows use the same amplitude scale
    bin_edges = compute_bin_edges(
        signal=data,
        sampling_rate=sampling_rate,
        n_bins=N_BINS  # configurable
    )

    print("Bin edges computed for HF1, HF2, LF bands")
    print(f"Number of bins: {N_BINS}")
    print()

    # STEP 4: Extract Features

    print_divider()
    print("STEP 4: Extracting features from windows")
    print_divider()

    # Generate windows using your existing window_generator
    windows = list(window_generator(
        data=data,
        td=TD,
        delta_t=DELTA_T,
        sampling_rate=sampling_rate,
        gaps=gaps,
        verbose=True
    ))

    print(f"Generated {len(windows)} valid windows")

    if len(windows) == 0:
        print("ERROR: No valid windows generated!")
        return False

    # Extract features from all windows
    feature_matrix, feature_names = extract_features_batch(
        windows=windows,
        sampling_rate=sampling_rate,
        bin_edges=bin_edges
    )

    if feature_names is None:
        print("ERROR: Feature extraction failed!")
        return False

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features per window: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print()

    # STEP 5: Save Results

    print_divider()
    print("STEP 5: Saving results")
    print_divider()

    # Get station info from the mseed file (not hardcoded!)
    st = read(input_file, headonly=True)
    station_name = st[0].stats.station
    network = st[0].stats.network
    data_start = min(tr.stats.starttime for tr in st)

    DATA_START_TIME = str(data_start)
    recording_date = data_start.strftime("%Y%m%d")

    print(f"Station: {network}.{station_name}")
    print(f"Recording start: {DATA_START_TIME}")

    # Create output folder: station_date
    output_dir = Path(f"data/processed/{station_name}_{recording_date}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrix
    features_file = output_dir / "features.npy"
    np.save(features_file, feature_matrix)
    print(f"Saved features: {features_file}")

    # Save feature names
    names_file = output_dir / "feature_names.txt"
    with open(names_file, 'w', encoding="utf-8") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"Saved feature names: {names_file}")

    # Save window timestamps
    timestamps = np.array([[w.start_hours, w.end_hours] for w in windows])
    timestamps_file = output_dir / "timestamps.npy"
    np.save(timestamps_file, timestamps)
    print(f"Saved timestamps: {timestamps_file}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metadata/config for reproducibility
    config = {
        'input_file': input_file,
        'station': station_name,
        'data_start_time': DATA_START_TIME,
        'sampling_rate': sampling_rate,
        'n_channels': n_channels,
        'duration_hours': duration_hours,
        'window_duration_hours': TD,
        'step_size_hours': DELTA_T,
        'n_bins': N_BINS,
        'n_windows': len(windows),
        'n_features': len(feature_names),
        'created': run_timestamp,
    }

    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_file}")

    print()

    # STEP 6: Generate Labels

    print_divider()
    print("STEP 6: Generating labels from earthquake catalog")
    print_divider()

    # Auto-fetch catalog if it doesn't exist
    catalog_dir = Path("data/catalogs")
    catalog_file = catalog_dir / f"catalog_{station_name}_{recording_date}.csv"

    if not catalog_file.exists():
        print("Catalog not found, fetching from USGS...")
        try:
            fetch_catalog(
                mseed_path=input_file,
                output_dir=str(catalog_dir),
                radius_km=500,
                min_magnitude=MIN_MAGNITUDE
            )
        except (requests.RequestException, OSError, ValueError) as e:
            print(f"ERROR: Failed to fetch catalog: {e}")
            return False

    CATALOG_FILE = str(catalog_file)

    # Load catalog
    catalog = load_catalog(CATALOG_FILE, DATA_START_TIME, MIN_MAGNITUDE)
    print(f"Loaded {len(catalog)} earthquakes (M >= {MIN_MAGNITUDE})")

    # Count earthquakes during recording period
    in_range: DataFrame = catalog[
        (catalog['hours_from_start'] >= 0) &
        (catalog['hours_from_start'] <= duration_hours)
    ]
    print(f"Earthquakes during recording: {len(in_range)}")

    # Generate labels
    labels = generate_labels(windows, catalog, LOOKAHEAD_HOURS)

    # Print stats
    stats = get_label_stats(labels)
    print("\nLabel distribution:")
    print(f"Total windows: {stats['total']}")
    print(
        f"Positive (EQ coming): {stats['positive']} ({stats['positive_pct']:.1f}%)")
    print(f"Negative (no EQ): {stats['negative']}")

    # Save labels
    labels_file = output_dir / "labels.npy"
    np.save(labels_file, labels)
    print(f"\nSaved labels: {labels_file}")

    # Save label config
    label_config = {
        'catalog_file': CATALOG_FILE,
        'lookahead_hours': LOOKAHEAD_HOURS,
        'min_magnitude': MIN_MAGNITUDE,
        'earthquakes_in_range': len(in_range),
        'positive_windows': int(stats['positive']),
        'negative_windows': int(stats['negative']),
    }

    label_config_file = output_dir / "label_config.json"
    with open(label_config_file, 'w', encoding="utf-8") as f:
        json.dump(label_config, f, indent=2)
    print(f"Saved label config: {label_config_file}")

    # Save processed catalog with hours_from_start

    processed_catalog_file = output_dir / "catalog_processed.csv"
    catalog.to_csv(processed_catalog_file, index=False)
    print(f"Saved processed catalog: {processed_catalog_file}")

    # SUMMARY

    print()
    print_divider()
    print("Pipeline complete!")
    print_divider()
    print(f"\nAll outputs saved to: {output_dir}/")
    print()
    print("Files created:")

    for file in sorted(output_dir.iterdir()):
        print(file.name)

    return True


def main() -> None:
    """Parse arguments and process mseed file(s)."""
    parser = argparse.ArgumentParser(
        description="Extract features from seismic data files"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/raw",
        help="Path to mseed file or directory (default: data/raw)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        mseed_files = sorted(input_path.glob("*.mseed"))
        print(f"Found {len(mseed_files)} mseed files in {input_path}\n")

        if len(mseed_files) == 0:
            print("No .mseed files found!")
            return

        succeeded = 0
        failed = 0

        for i, mseed_file in enumerate(mseed_files, 1):
            print(f"\n{'#'*60}")
            print(f"# FILE {i}/{len(mseed_files)}: {mseed_file.name}")
            print(f"{'#'*60}\n")

            if process_file(str(mseed_file)):
                succeeded += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Succeeded: {succeeded}")
        print(f"Failed: {failed}")
        print(f"Total: {len(mseed_files)}")

    else:
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            return
        process_file(str(input_path))


if __name__ == "__main__":
    main()
