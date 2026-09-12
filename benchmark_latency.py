"""
Measure actual preprocessing time, model inference time, and total time on
the live-streaming code path (raw PCM16 bytes -> detector.predict), not on
file-loading -- that's what matters for the "sub-500ms CPU inference"
target. No number here is a claim until you run this yourself and read the
printed output; nothing is hard-coded.

Usage:
    python -m models.deepfake.benchmark_latency [--n 100] [--wav path/to/real_or_synth.wav]

Without --wav, generates deterministic pseudo-random noise windows just to
measure raw compute cost (this tells you nothing about accuracy, only
speed -- use eval.py for accuracy on real data).
"""
from __future__ import annotations

import argparse
import struct
import time

import numpy as np

from .detector import AcousticDeepfakeDetector
from .preprocess import pcm16_bytes_to_waveform

WINDOW_SAMPLES = 32000  # 2.0s at 16kHz, matches the backend's BUFFER_THRESHOLD


def _make_noise_window(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(WINDOW_SAMPLES) * 3000).astype(np.int16)
    return samples.tobytes()


def _load_wav_as_pcm16_bytes(path: str) -> bytes:
    from .preprocess import wav_file_to_waveform

    waveform, _ = wav_file_to_waveform(path)
    if waveform.shape[0] > WINDOW_SAMPLES:
        waveform = waveform[:WINDOW_SAMPLES]
    elif waveform.shape[0] < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.shape[0]))
    pcm = (waveform * 32767.0).astype(np.int16)
    return pcm.tobytes()


def run(n: int, wav_path: str | None) -> None:
    detector = AcousticDeepfakeDetector()
    if detector._adapter is None:
        raise SystemExit(f"Model failed to load: {detector._load_error}")

    if wav_path:
        windows = [_load_wav_as_pcm16_bytes(wav_path) for _ in range(n)]
    else:
        windows = [_make_noise_window(i) for i in range(n)]

    # Warm-up: first call includes lazy CUDA/threading init overhead that
    # isn't representative of steady-state latency.
    detector.predict(windows[0])

    preprocess_times = []
    total_times = []

    for window in windows:
        t0 = time.perf_counter()
        pcm16_bytes_to_waveform(window)
        t1 = time.perf_counter()
        detector.predict(window)
        t2 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)

    def stats(name, values):
        values = sorted(values)
        n_ = len(values)
        print(
            f"{name:12s} mean={sum(values)/n_:7.2f}ms  "
            f"p50={values[n_//2]:7.2f}ms  "
            f"p95={values[int(n_*0.95)]:7.2f}ms  "
            f"max={values[-1]:7.2f}ms"
        )

    print(f"Ran {n} windows ({'from ' + wav_path if wav_path else 'synthetic noise'})")
    stats("preprocess", preprocess_times)
    stats("total", total_times)
    print()
    print(
        "Note: 'total' includes preprocessing + model inference "
        "(it does not separately isolate model-only time here -- "
        "subtract preprocess from total for a rough model-only estimate)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--wav", type=str, default=None)
    args = parser.parse_args()
    run(args.n, args.wav)
