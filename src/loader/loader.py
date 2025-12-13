"""
loader.py: Load seismic data from MiniSEED files

What this does:
    Takes a file path and then returns numpy array of waveform data

The data comes out as shape (n_channels, n_samples):
    - n_channels = number of sensors/channels in the file
    - n_samples = number of data points (time steps)
"""

from typing import Dict, Optional, List, Tuple
import numpy as np


def load_file(filepath: str,
              channels: Optional[List[int]] = None,
              max_length_diff: int = 100
              ) -> Tuple[np.ndarray, Dict[str, int | str | float]]:
    """
    Load a seismic data file.

    Args:
        filepath: Path to the .mseed file
        channels: Optional list of channel indices to load (None = all)
                  This saves memory for large files.
        max_length_diff: Maximum allowed difference in samples between channels.
                     If exceeded, raises ValueError. Default 100.

    Returns:
        data: numpy array of shape (n_channels, n_samples)
        metadata: dict with info like sampling_rate, duration, etc.

    Raises:
        ValueError: If any traces differ too much in terms of length of data
    """
    pass


def get_file_info(filepath: str) -> Dict[str, str | int]:
    """
    Get info about a file WITHOUT loading all the data.
    Useful for checking how big a file is before loading.
    """
    pass
