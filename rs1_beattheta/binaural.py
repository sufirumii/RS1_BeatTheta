"""Step 5b - Measuring the binaural beat in finished audio.

The beat itself is created by plain signal processing inside the model's generate() method
(left and right sine tones whose frequencies differ by exactly beat_hz). This module only measures it,
so the claim "the beat is exact" can be checked on real output.
"""
import numpy as np
import scipy.signal as sps


def _instantaneous_frequency(x, sr):
    return np.diff(np.unwrap(np.angle(sps.hilbert(x)))) * sr / (2 * np.pi)


def measure_beat(audio, sr, carrier_hz, expected_beat_hz, trim_s=1.0):
    """Return the measured beat in Hz (right channel frequency minus left channel frequency)."""
    half = expected_beat_hz / 2 + 4.0
    sos = sps.butter(4, [carrier_hz - half, carrier_hz + half], "bandpass", fs=sr, output="sos")
    left = sps.sosfiltfilt(sos, audio[:, 0].astype(np.float64))
    right = sps.sosfiltfilt(sos, audio[:, 1].astype(np.float64))
    diff = _instantaneous_frequency(right, sr) - _instantaneous_frequency(left, sr)
    trim = min(int(trim_s * sr), len(diff) // 4)
    return float(np.mean(diff[trim:len(diff) - trim]))
