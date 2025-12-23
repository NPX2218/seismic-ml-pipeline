"""
combine_datasets.py: Combine all processed datasets for ML training.

This script combines feature matrices and labels from all processed
station folders into single arrays ready for machine learning.

Usage:
    python combine_datasets.py

Outputs:
    data/ml_ready/X.npy - Combined feature matrix (n_samples, n_features)
    data/ml_ready/y.npy - Combined labels (n_samples,)
"""

from pathlib import Path
from typing import Any, List

import numpy as np
from numpy.typing import NDArray


def combine_datasets(
    processed_dir: str = "data/processed",
    output_dir: str = "data/ml_ready"
) -> None:
    """
    Combine all processed datasets into single arrays for ML training.

    Args:
        processed_dir: Directory containing processed station folders
        output_dir: Directory to save combined arrays
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)

    all_features: List[NDArray[np.floating[Any]]] = []
    all_labels: List[NDArray[np.integer[Any]]] = []

    # Iterate over directories only, skip hidden files like .DS_Store
    folders = sorted([
        f for f in processed_path.iterdir()
        if f.is_dir() and not f.name.startswith('.')
    ])

    if len(folders) == 0:
        print(f"No processed folders found in {processed_dir}")
        return

    print(f"Found {len(folders)} processed folders")
    print()

    for folder in folders:
        features_file = folder / "features.npy"
        labels_file = folder / "labels.npy"

        # Skip folders that don't have both files
        if not features_file.exists() or not labels_file.exists():
            print(f"  Skipping {folder.name} (missing files)")
            continue

        features = np.load(features_file)
        labels = np.load(labels_file)

        all_features.append(features)
        all_labels.append(labels)

        print(f"  {folder.name}: {features.shape[0]} windows")

    if len(all_features) == 0:
        print("No valid datasets found!")
        return

    # Combine all arrays
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save combined arrays
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)

    # Print summary
    print()
    print("=" * 50)
    print("COMBINED DATASET")
    print("=" * 50)
    print(f"Total samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    print(
        f"Positive labels: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
    print(
        f"Negative labels: {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print()
    print("Saved to:")
    print(f"{output_path / 'X.npy'}")
    print(f"{output_path / 'y.npy'}")


def main() -> None:
    """Main entry point."""
    combine_datasets()


if __name__ == "__main__":
    main()
