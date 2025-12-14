from obspy import read


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
        print(f"Trace {i}:")
        print(f"  Station: {stats.station}")
        print(f"  Channel: {stats.channel}")
        print(f"  Start: {stats.starttime}")
        print(f"  End: {stats.endtime}")
        print(f"  Samples: {stats.npts:,}")
        print()
