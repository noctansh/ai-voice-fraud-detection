"""
Audio preprocessing and validation for the acoustic deepfake detector.

Upstream contract (see app/main.py in the backend repo):
    - audio arrives as RAW PCM16 bytes (no WAV/RIFF header)
    - 16 kHz, mono
    - ~2.0 second sliding windows (64000 bytes = 32000 samples)

This module does NOT resample or remix channels silently for the raw-bytes
path, because we have no header telling us the real sample rate/channel
count -- guessing would corrupt the audio. It validates the assumption and
raises a clear error if the byte count is inconsistent with 16-bit mono
samples. A separate helper is provided for the case where you *do* have a
WAV file (e.g. loading test data from real/ and synthetic/ folders), where
the header is trusted and mismatches are actually fixed (resample / to-mono).
"""
from __future__ import annotations

import numpy as np

TARGET_SAMPLE_RATE = 16000
MIN_SAMPLES = 800  # 50ms at 16kHz -- below this, detection is not meaningful
SILENCE_RMS_THRESHOLD = 1e-4  # on a [-1, 1]-normalized signal


class AudioValidationError(ValueError):
    """Raised when incoming audio cannot be safely interpreted."""


def pcm16_bytes_to_waveform(audio_bytes: bytes) -> tuple[np.ndarray, list[str]]:
    """
    Convert raw 16-bit PCM mono bytes (as sent by the streaming backend) into
    a float32 waveform normalized to [-1, 1].

    Returns (waveform, flags). Flags are non-fatal quality observations
    (e.g. "low_energy"), not proof of anything about the speaker's identity
    or authenticity -- poor audio quality must never be treated as evidence
    of a synthetic voice.

    Raises AudioValidationError for genuinely unusable input (corrupt,
    empty, or too short to run the model on at all).
    """
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise AudioValidationError(
            f"expected bytes-like audio, got {type(audio_bytes).__name__}"
        )

    if len(audio_bytes) == 0:
        raise AudioValidationError("received empty audio buffer")

    if len(audio_bytes) % 2 != 0:
        raise AudioValidationError(
            "byte count is odd -- not valid 16-bit PCM (corrupt or "
            "truncated buffer)"
        )

    try:
        pcm = np.frombuffer(audio_bytes, dtype="<i2")
    except ValueError as exc:
        raise AudioValidationError(f"could not parse buffer as PCM16: {exc}") from exc

    n_samples = pcm.shape[0]
    if n_samples < MIN_SAMPLES:
        duration_ms = 1000 * n_samples / TARGET_SAMPLE_RATE
        raise AudioValidationError(
            f"audio too short to analyze: {n_samples} samples "
            f"(~{duration_ms:.0f}ms) < minimum {MIN_SAMPLES}"
        )

    waveform = pcm.astype(np.float32) / 32768.0

    flags: list[str] = []

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < SILENCE_RMS_THRESHOLD:
        flags.append("low_energy")

    if np.isnan(waveform).any() or np.isinf(waveform).any():
        raise AudioValidationError("audio contains NaN/Inf samples (corrupt buffer)")

    duration_s = n_samples / TARGET_SAMPLE_RATE
    if duration_s < 1.0:
        flags.append("short_window")

    peak = float(np.max(np.abs(waveform)))
    if peak >= 0.999:
        flags.append("possible_clipping")

    return waveform, flags


def wav_file_to_waveform(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load a WAV file (used for local test-set evaluation, not the live
    streaming path) and coerce it to 16kHz mono float32, resampling /
    downmixing when the file's header says it's something else. Flags any
    conversion performed so it's visible in eval reports.

    Requires soundfile (already a dependency of the AASIST reference code).
    """
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    flags: list[str] = []

    if data.shape[1] > 1:
        data = data.mean(axis=1)
        flags.append("downmixed_to_mono")
    else:
        data = data[:, 0]

    if sr != TARGET_SAMPLE_RATE:
        data = _resample_linear(data, sr, TARGET_SAMPLE_RATE)
        flags.append(f"resampled_from_{sr}hz")

    if data.shape[0] < MIN_SAMPLES:
        raise AudioValidationError(
            f"{path}: audio too short to analyze ({data.shape[0]} samples)"
        )

    rms = float(np.sqrt(np.mean(np.square(data))))
    if rms < SILENCE_RMS_THRESHOLD:
        flags.append("low_energy")

    return data.astype(np.float32), flags


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Simple linear-interpolation resampler. Good enough for test-set loading
    where files are already close to 16kHz; NOT a substitute for a proper
    resampling library in a real product. Kept dependency-free on purpose.
    """
    if sr_in == sr_out:
        return x
    duration = x.shape[0] / sr_in
    n_out = int(round(duration * sr_out))
    x_times = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    out_times = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(out_times, x_times, x).astype(np.float32)
