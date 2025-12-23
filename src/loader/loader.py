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
from obspy import Trace, read
from obspy import Stream


def merge_traces(st: Stream, verbose: bool = True) -> tuple[Stream, list[tuple[float, float]]]:
    """
    Merge fragmented traces from the same channel.

    Returns:
        Merged ObsPy Stream
        List of gap locations as (start_seconds, end_seconds) from beginning of data
    """
    if verbose:
        print(f"Before merge: {len(st)} traces")

    # Track gap locations
    gaps: list[tuple[float, float]] = []

    # Find the earliest start time (reference point)
    earliest_start = min(trace.stats.starttime for trace in st)

    channel_groups: Dict[str, list[Trace]] = {}

    for trace in st:
        channel_id = f"{trace.stats.station}.{trace.stats.channel}"
        channel_groups.setdefault(channel_id, []).append(trace)

    if verbose:
        print(
            f"Found {len(channel_groups)} unique channels: {list(channel_groups.keys())}")

    for channel_id, traces in channel_groups.items():
        sorted_traces = sorted(traces, key=lambda t: t.stats.starttime)

        if verbose and len(sorted_traces) > 1:
            print(
                f"\nMerging {channel_id} ({len(sorted_traces)} fragments):")

        for i, trace in enumerate(sorted_traces):
            if verbose and len(sorted_traces) > 1:
                print(
                    f"Fragment {i}: {trace.stats.starttime} → {trace.stats.endtime}")

            if i < len(sorted_traces) - 1:
                current_end = trace.stats.endtime
                next_start = sorted_traces[i + 1].stats.starttime
                gap_seconds = next_start - current_end

                if gap_seconds < 0:
                    raise ValueError(
                        f"Overlap detected in {channel_id}! "
                        f"Trace {i} ends at {current_end}, "
                        f"Trace {i+1} starts at {next_start} "
                        f"(overlap: {-gap_seconds:.2f} seconds)"
                    )

                if gap_seconds > 0:
                    # Record gap position relative to start of data
                    gap_start = current_end - earliest_start
                    gap_end = next_start - earliest_start
                    gaps.append((float(gap_start), float(gap_end)))

                    if verbose:
                        print(
                            f"Gap to next: {gap_seconds:.2f} seconds (recorded)")

    # Use interpolation for merge (smoother than zeros, but we'll skip these windows anyway)
    st.merge(method=1, interpolation_samples=0, fill_value='interpolate')

    if verbose:
        print(f"\nAfter merge: {len(st)} traces")
        print(f"Recorded {len(gaps)} gaps to skip during windowing")
        for trace in st:
            duration_hours = trace.stats.npts / trace.stats.sampling_rate / 3600
            print(
                f"{trace.stats.station}.{trace.stats.channel}: {trace.stats.npts:,} samples ({duration_hours:.2f} hours)")

    return st, gaps


def load_file(filepath: str,
              channels: Optional[List[int]] = None,
              max_length_diff: int = 100,
              max_gap_percent: float = 5.0,
              verbose: bool = True

              ) -> Tuple[np.ndarray, Dict[str, int | str | float]]:
    """
    Load a seismic data file.
    Args:
        filepath: Path to the .mseed file
        channels: Optional list of channel indices to load (None = all)
                  This saves memory for large files.
        max_length_diff: Maximum allowed difference in samples between channels. If exceeded, raises ValueError. Default 100.
        max_gap_percentage: The maxmimum percentage of gap time over the total time for data to be processed further.
        verbose: If True, print details about merging process.
    Returns:
        data: numpy array of shape (n_channels, n_samples)
        metadata: dict with info like sampling_rate, duration, etc.
    Raises:
        ValueError: If any traces differ too much in terms of length of data
        ValueError: If overlapping traces are detected
    """

    # The structure of the file is like [Trace0, Trace1, Trace2]
    st = read(filepath)
    total_channels_before_merge = len(st)

    st, gaps = merge_traces(st, verbose=verbose)

    total_channels = len(st)

    total_duration = st[0].stats.npts / st[0].stats.sampling_rate
    total_gap_seconds = sum(gap_end - gap_start for gap_start, gap_end in gaps)
    gap_percent = (total_gap_seconds / total_duration) * 100

    if verbose:
        print(
            f"Total gap time: {total_gap_seconds:.2f} seconds ({gap_percent:.2f}% of recording)")

    if gap_percent > max_gap_percent:
        raise ValueError(
            f"Too many gaps in data! "
            f"Gap time: {total_gap_seconds:.1f}s ({gap_percent:.1f}%) "
            f"exceeds threshold of {max_gap_percent}%. "
            f"Data may be too fragmented for reliable ML training."
        )

    # Select channels if specified
    if channels is not None:
        # Filter to only requested channels
        selected_traces = [st[i] for i in channels if i < total_channels]
    else:
        selected_traces = list(st)

    # Extract data from selected traces
    data_list = []
    for trace in selected_traces:
        # Converts the data for each trace from int to float
        data_list.append(trace.data.astype(np.float64))

    lengths = [len(d) for d in data_list]
    if max(lengths) - min(lengths) > max_length_diff:
        raise ValueError(f"Channel lengths vary too much: {lengths}")

    min_length = min(lengths)

    # Makes sure all the data is up to only the min_length so they match in size
    data = np.array([d[:min_length] for d in data_list])

    n_channels = len(data_list)

    # Metadata
    sampling_rate = st[0].stats.sampling_rate

    metadata = {
        'filepath': filepath,
        'n_channels': n_channels,
        'total_traces_in_file': total_channels_before_merge,
        'n_samples': min_length,
        'sampling_rate': sampling_rate,
        'duration_seconds': min_length / sampling_rate,
        'duration_hours': min_length / sampling_rate / 3600,
        'gaps': gaps,

    }

    return data, metadata


def get_file_info(filepath: str) -> Dict[str, str | int]:
    """
    Get info about a file WITHOUT loading all the data.
    Useful for checking how big a file is before loading.
    """

    st = read(filepath, headonly=True)  # Only read headers, not data

    return {
        'filepath': filepath,
        # Number of separate directions (usually 3 for x, y, z)
        'n_channels': len(st),
        # How many measurements per second
        'sampling_rate': st[0].stats.sampling_rate,
        'n_samples': st[0].stats.npts,  # Total number of data points recorded
        'duration_hours': st[0].stats.npts / st[0].stats.sampling_rate / 3600,
    }
