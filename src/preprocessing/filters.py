"""
filters.py — Bandpass filtering to decompose signals

What this does:
    Takes raw signal and splits into 3 frequency bands

The 3 bands:
    HF1 (5-25 Hz): Rapid ground vibrations
    HF2 (0.1-5 Hz): Medium-speed ground motion
    LF (residual): Slow drift (whatever's left)
"""

from typing import Tuple, Dict
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data: np.ndarray,
                    lowcut: float,
                    highcut: float,
                    sampling_rate: float,
                    order: int = 4) -> np.ndarray:
    """
    Apply a bandpass filter to keep only frequencies between lowcut and highcut.
    
    Args:
        data: Input signal (1D array for one channel)
        lowcut: Lower frequency bound (Hz) — frequencies below this are removed
        highcut: Upper frequency bound (Hz) — frequencies higher than this are removed
        sampling_rate: Samples per second (Hz)
        order: Filter sharpness (4 is a good default)
    
    Returns:
        Filtered signal (same shape as input)
    """

    # butterworth needs frequencies as a fraction of Nyquist
    nyquist = 0.5 * sampling_rate

    low = lowcut / nyquist
    high = highcut / nyquist

    # Turn to valid range (0-1, exclusive)
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.001, 0.999)

    b, a = butter(order, [low, high], btype='band') # returns filter coefficients (b, a)

    # Preserves exact timing
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def decompose_signal(data: np.ndarray, sampling_rate: float) -> Dict[str, np.ndarray]:
    """
    Decompose a signal into three frequency bands: HF1, HF2, LF
    
    Args:
        data: Raw signal (1D array)
        sampling_rate: Samples per second (Hz)
    
    Returns:
        Dict w/ keys 'HF1', 'HF2', 'LF', each containing filtered signal
    """

    hf1 = bandpass_filter(data, lowcut=5.0, highcut=25.0, sampling_rate=sampling_rate)

    hf2 = bandpass_filter(data, lowcut=0.1, highcut=5.0, sampling_rate=sampling_rate)

    lf = data - hf1 - hf2

    return {
        'HF1': hf1,
        'HF2': hf2,
        'LF': lf
    }
