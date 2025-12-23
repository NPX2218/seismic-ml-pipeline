"""
features.py: Extract histogram-based amplitude features

This file extracts two features from each frequency band using
histogram analysis of log-transformed amplitudes:

1. Center of mass: weighted average of log-amplitudes
2. Spread: weighted mean absolute deviation from center

Note: Bin edges must be computed ONCE from the entire signal,
      then reused for all windows. This ensures consistency.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from src.preprocessing.filters import decompose_signal
from src.preprocessing.windows import Window


# Small constant to avoid log(0)
EPSILON: float = 1e-10


@dataclass
class BinEdges:
    """
    Container for precomputed bin edges.

    Bin edges are computed ONCE from the entire signal,
    then reused for all windows to ensure consistency.
    """
    HF1: np.ndarray  # shape: (n_bins + 1,)
    HF2: np.ndarray
    LF: np.ndarray
    midpoints_HF1: np.ndarray  # shape: (n_bins,)
    midpoints_HF2: np.ndarray
    midpoints_LF: np.ndarray


def compute_bin_edges(
    signal: np.ndarray,
    sampling_rate: float,
    n_bins: int
) -> BinEdges:
    """
    Compute bin edges from the ENTIRE signal (before windowing).

    This must be called ONCE on the full signal, and the resulting
    BinEdges object is passed to all subsequent feature extractions.

    Args:
        signal: Full signal, shape (n_channels, n_samples) or (n_samples,)
        sampling_rate: Samples per second (Hz)
        n_bins: Number of histogram bins

    Returns:
        BinEdges object containing edges and midpoints for each band
    """
    # Handle 1D input
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    # Use first channel to compute bin edges
    channel_signal = signal[0]

    # Decompose into frequency bands
    bands = decompose_signal(channel_signal, sampling_rate)

    edges_dict: Dict[str, np.ndarray] = {}
    midpoints_dict: Dict[str, np.ndarray] = {}

    for band_name in ['HF1', 'HF2', 'LF']:
        band_signal = bands[band_name]

        # Transform: z = log|x|
        z = np.log(np.abs(band_signal) + EPSILON)

        # Compute bin edges from min to max
        z_min = np.min(z)
        z_max = np.max(z)

        # n_bins + 1 edges define n_bins bins
        edges = np.linspace(z_min, z_max, n_bins + 1)

        # Compute bin midpoints
        midpoints = (edges[:-1] + edges[1:]) / 2

        edges_dict[band_name] = edges
        midpoints_dict[band_name] = midpoints

    return BinEdges(
        HF1=edges_dict['HF1'],
        HF2=edges_dict['HF2'],
        LF=edges_dict['LF'],
        midpoints_HF1=midpoints_dict['HF1'],
        midpoints_HF2=midpoints_dict['HF2'],
        midpoints_LF=midpoints_dict['LF']
    )


def compute_histogram_features(
    signal: np.ndarray,
    bin_edges: np.ndarray,
    midpoints: np.ndarray
) -> Tuple[float, float]:
    """
    Compute center of mass and spread from histogram of log-amplitudes.

    Args:
        signal: 1D array of the filtered waveform (one band)
        bin_edges: Precomputed bin edges, shape (n_bins + 1)
        midpoints: Precomputed bin midpoints, shape (n_bins)

    Returns:
        center_of_mass: Weighted average of log-amplitudes
        spread: Weighted mean absolute deviation
    """
    # z = log|x|
    z = np.log(np.abs(signal) + EPSILON)

    # Use precomputed edges for consistency across windows
    counts, _ = np.histogram(z, bins=bin_edges)

    total_count = np.sum(counts)

    if total_count == 0:
        return 0.0, 0.0

    # Normalized: p_i = counts[i] / total_count
    center_of_mass = np.sum(midpoints * counts) / total_count

    absolute_deviations = np.abs(midpoints - center_of_mass)
    spread = np.sum(absolute_deviations * counts) / total_count

    return float(center_of_mass), float(spread)


def extract_features(
    window: Window,
    sampling_rate: float,
    bin_edges: BinEdges
) -> Dict[str, float]:
    """
    Extract histogram-based features from a window.

    Args:
        window: A Window object containing the data
        sampling_rate: Samples per second (Hz)
        bin_edges: Precomputed BinEdges object (from compute_bin_edges)

    Returns:
        Dictionary of feature_name to value
    """
    features: Dict[str, float] = {}
    data = window.data
    n_channels = data.shape[0]

    # Map band names to their edges and midpoints
    edges_map = {
        'HF1': (bin_edges.HF1, bin_edges.midpoints_HF1),
        'HF2': (bin_edges.HF2, bin_edges.midpoints_HF2),
        'LF': (bin_edges.LF, bin_edges.midpoints_LF),
    }

    for ch in range(n_channels):
        channel_signal = data[ch]

        # Decompose into frequency bands
        bands = decompose_signal(channel_signal, sampling_rate)

        for band_name in ['HF1', 'HF2', 'LF']:
            band_signal = bands[band_name]
            edges, midpoints = edges_map[band_name]

            ym, spread = compute_histogram_features(
                signal=band_signal,
                bin_edges=edges,
                midpoints=midpoints
            )

            features[f'ch{ch}_{band_name}_ym'] = ym
            features[f'ch{ch}_{band_name}_spread'] = spread

    return features


def extract_features_batch(
    windows: List[Window],
    sampling_rate: float,
    bin_edges: BinEdges
) -> Tuple[np.ndarray, List[str] | None]:
    """
    Extract features from multiple windows.

    Args:
        windows: List of Window objects
        sampling_rate: Samples per second (Hz)
        bin_edges: Precomputed BinEdges object

    Returns:
        feature_matrix: Shape (n_windows, n_features)
        feature_names: List of feature names
    """
    all_features: List[np.ndarray] = []
    feature_names: List[str] | None = None

    for window in windows:
        feat_dict = extract_features(window, sampling_rate, bin_edges)

        if feature_names is None:
            feature_names = list(feat_dict.keys())

        feat_array = np.array(list(feat_dict.values()))
        all_features.append(feat_array)

    feature_matrix = np.array(all_features)
    return feature_matrix, feature_names
