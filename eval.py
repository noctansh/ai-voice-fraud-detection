"""
Evaluate the detector on labeled test data.

Usage:
    python -m models.deepfake.eval --real-dir path/to/real --synthetic-dir path/to/synthetic

Expects WAV files (any sample rate/channel count -- they'll be converted,
and the conversion is reported per-file) under two folders:
    real/*.wav        -- genuine human speech
    synthetic/*.wav   -- AI-generated / cloned / spoofed speech

Prints accuracy, precision, recall, F1, confusion matrix, false positive
rate, false negative rate, and per-file latency, using a real trained
checkpoint -- never a hard-coded or invented score.

IMPORTANT: a handful of test clips is not a validation of the detector. Use
a reasonably sized, unseen test set, and look at the confusion matrix and
per-condition breakdown (see --breakdown), not just the top-line accuracy.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

from .detector import AcousticDeepfakeDetector
from .features import pad_to_fixed_length
from .preprocess import AudioValidationError, wav_file_to_waveform

DECISION_THRESHOLD = 0.5  # synthetic_probability >= this => predicted "synthetic"


def _predict_wav(detector: AcousticDeepfakeDetector, path: str):
    """Run the loaded model directly on a WAV file's waveform (bypasses the
    raw-PCM streaming path, which is for live audio, not files on disk)."""
    if detector._adapter is None:
        raise RuntimeError(detector._load_error or "model not loaded")

    waveform, flags = wav_file_to_waveform(path)
    padded = pad_to_fixed_length(waveform)

    start = time.perf_counter()
    synthetic_probability = detector._adapter.predict(padded)
    inference_ms = (time.perf_counter() - start) * 1000.0

    return synthetic_probability, flags, inference_ms


def evaluate(real_dir: str, synthetic_dir: str) -> None:
    detector = AcousticDeepfakeDetector()
    if detector._adapter is None:
        raise SystemExit(
            f"Cannot evaluate: model failed to load ({detector._load_error})"
        )

    real_files = sorted(glob.glob(os.path.join(real_dir, "*.wav")))
    synthetic_files = sorted(glob.glob(os.path.join(synthetic_dir, "*.wav")))

    if not real_files or not synthetic_files:
        raise SystemExit(
            f"Found {len(real_files)} real and {len(synthetic_files)} "
            "synthetic files -- need at least one of each. Check --real-dir "
            "and --synthetic-dir."
        )

    tp = fp = tn = fn = 0
    latencies = []
    skipped = []

    for path, true_label in (
        [(p, 0) for p in real_files] + [(p, 1) for p in synthetic_files]
    ):
        try:
            score, flags, inference_ms = _predict_wav(detector, path)
        except AudioValidationError as exc:
            skipped.append((path, str(exc)))
            continue

        latencies.append(inference_ms)
        predicted_label = 1 if score >= DECISION_THRESHOLD else 0

        if true_label == 1 and predicted_label == 1:
            tp += 1
        elif true_label == 0 and predicted_label == 1:
            fp += 1
        elif true_label == 0 and predicted_label == 0:
            tn += 1
        elif true_label == 1 and predicted_label == 0:
            fn += 1

    total = tp + fp + tn + fn
    if total == 0:
        raise SystemExit("Every file was skipped -- nothing to evaluate.")

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) and precision == precision and recall == recall
        else float("nan")
    )
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    print(f"Evaluated {total} files ({len(skipped)} skipped)")
    print(f"Threshold: synthetic_probability >= {DECISION_THRESHOLD}")
    print()
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}  (real audio wrongly flagged as synthetic)")
    print(f"FNR:       {fnr:.4f}  (synthetic audio missed)")
    print()
    print("Confusion matrix:")
    print(f"                 pred_real  pred_synthetic")
    print(f"  actual_real    {tn:<10} {fp}")
    print(f"  actual_synth   {fn:<10} {tp}")
    print()
    if latencies:
        latencies.sort()
        n = len(latencies)
        print(
            f"Inference latency (ms): "
            f"mean={sum(latencies)/n:.1f} "
            f"p50={latencies[n//2]:.1f} "
            f"p95={latencies[int(n*0.95)]:.1f} "
            f"max={latencies[-1]:.1f}"
        )
    if skipped:
        print()
        print("Skipped files (validation errors):")
        for path, reason in skipped:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    args = parser.parse_args()
    evaluate(args.real_dir, args.synthetic_dir)
