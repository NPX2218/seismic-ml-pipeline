"""
windows.py — Create sliding windows from continuous data

What this does:
    Takes long signal into overlapping chunks one at a time
    Skips windows w/ gaps

Parameters:
    td = Window duration (how much data to look at) - 2 hours
    delta_t = Step size (how far to slide each time) - 0.5 hours
    gaps = List of (start_seconds, end_seconds) tuples to avoid
"""

from typing import Generator, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Window:
    """Container for a window of data"""
    data: np.ndarray
    index: int
    start_hours: float
    end_hours: float


def _window_overlaps_gap(window_start_sec: float,
                         window_end_sec: float,
                         gaps: List[Tuple[float, float]]) -> bool:
    """
    Check if a window overlaps w/ any gap in the data.

    Two intervals [A, B] and [C, D] overlap if: A < D AND B > C

    Args:
        window_start_sec: Window start time 
        window_end_sec: Window end time  
        gaps: List of (gap_start_sec, gap_end_sec) tuples

    Returns:
        True if window overlaps any gap, False otherwise
    """
    for gap_start, gap_end in gaps:
        # Check for overlap
        if window_start_sec < gap_end and window_end_sec > gap_start:
            return True
    return False

def count_windows(n_samples: int,
                  td: float,
                  delta_t: float,
                  sampling_rate: float,
                  gaps: Optional[List[Tuple[float, float]]] = None) -> int:
    """
    Calculate how many valid windows (w/o gaps) will be created.
    
    Args:
        n_samples: Total samples in signal
        td: Window duration in hours
        delta_t: Step size in hours
        sampling_rate: Samples per second (Hz)
        gaps: Optional list of (start_sec, end_sec) gap tuples
        
    Returns:
        Number of valid windows (excluding those overlapping gaps)
    """
    # Convert hours to samples
    window_samples = int(td * 3600 * sampling_rate)
    step_samples = int(delta_t * 3600 * sampling_rate)

    if n_samples < window_samples:
        return 0

    if gaps is None or len(gaps) == 0:
        # No gaps, divide windows directly
        return (n_samples - window_samples) // step_samples + 1

    valid_count = 0
    start = 0

    while start + window_samples <= n_samples:
        end = start + window_samples

        # Convert to seconds for gap comparison
        start_sec = start / sampling_rate
        end_sec = end / sampling_rate

        if not _window_overlaps_gap(start_sec, end_sec, gaps):
            valid_count += 1

        start += step_samples

    return valid_count

def window_generator(data: np.ndarray,
                     td: float,
                     delta_t: float,
                     sampling_rate: float,
                     gaps: Optional[List[Tuple[float, float]]] = None,
                     verbose: bool = False) -> Generator[Window, None, None]:
    """
    Generate windows one at a time 
    Automatically skips windows that overlap w/ data gaps.

    Args:
        data: Input signal, shape (n_channels, n_samples)
        td: Window duration in hours
        delta_t: Step size in hours
        sampling_rate: Samples per second (Hz)
        gaps: Optional list of (start_seconds, end_seconds) tuples
        verbose: If True, print when windows are skipped

    Returns:
        Valid window objects
    """
    # nNormalize input shape by handling single and multi-channel input
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_channels, n_samples = data.shape

    if gaps is None:
        gaps = []

    # Convert hours to samples
    window_samples = int(td * 3600 * sampling_rate)
    step_samples = int(delta_t * 3600 * sampling_rate)

    window_index = 0  # valid windows
    position_index = 0  # all potential window positions
    start = 0
    skipped_count = 0

    while start + window_samples <= n_samples:
        end = start + window_samples

        # Convert to sec for gap comparison
        start_sec = start / sampling_rate
        end_sec = end / sampling_rate

        # Check if window overlaps any gap
        if _window_overlaps_gap(start_sec, end_sec, gaps):
            if verbose:
                print(f"  Skipping window at position {position_index} "
                      f"({start_sec/3600:.2f}h - {end_sec/3600:.2f}h): overlaps gap")
            skipped_count += 1
            start += step_samples
            position_index += 1
            continue  # Skip & move to next

        # Yield valid window (yield used due to generating)
        yield Window(
            data=data[:, start:end],  # view rather than copy
            index=window_index,
            start_hours=start_sec/3600,
            end_hours=end_sec/3600
        )

        start += step_samples
        window_index += 1
        position_index += 1

    if verbose and skipped_count > 0:
        print(f"\nWindow generation complete: {window_index} valid, {skipped_count} skipped gaps")
