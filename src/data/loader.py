import os
from pathlib import Path
from obspy import read


def load_files_from_data_folder() -> None:

    path = Path(__file__).parent / "raw"  # subfolder called "data"

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)

        load_file_from_path(filepath)
        return  # For testing we will only test one file


def load_file_from_path(path: str) -> None:
    st = read(path)
    print(f"Loading in Earthquake: {st[0].stats.starttime}")
