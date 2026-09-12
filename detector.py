"""
Public entry point for the acoustic deepfake detector.

    from models.deepfake.detector import AcousticDeepfakeDetector
    detector = AcousticDeepfakeDetector()
    result = await detector.predict_async(audio_bytes)

`result` schema (see README.md for the authoritative version):
    {
        "synthetic_probability": float in [0.0, 1.0] or None on error,
        "artifact_flags": list[str],
        "model_name": "AASIST-L",
        "inference_ms": float,
        "error": str | None,
    }

This module never invents a score. If the model can't load or the audio
fails validation, "error" is set and "synthetic_probability" is None --
callers must not treat None as 0.0 or as "safe".
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, TypedDict

from .features import pad_to_fixed_length
from .model_adapter import AASISTAdapter, ModelUnavailableError
from .preprocess import AudioValidationError, pcm16_bytes_to_waveform

MODEL_NAME = "AASIST-L"


class DetectionResult(TypedDict):
    synthetic_probability: Optional[float]
    artifact_flags: list
    model_name: str
    inference_ms: float
    error: Optional[str]


class AcousticDeepfakeDetector:
    def __init__(self):
        self._adapter: Optional[AASISTAdapter] = None
        self._load_error: Optional[str] = None
        try:
            self._adapter = AASISTAdapter()
        except ModelUnavailableError as exc:
            # Don't crash on import/construction -- record the error and
            # surface it on every prediction instead, per "don't pretend
            # it's a production detector if the checkpoint isn't there."
            self._load_error = str(exc)

    def predict(self, audio_bytes: bytes) -> DetectionResult:
        """Synchronous prediction, for local testing/scripts."""
        start = time.perf_counter()

        if self._adapter is None:
            return _error_result(self._load_error or "model not loaded", start)

        try:
            waveform, flags = pcm16_bytes_to_waveform(audio_bytes)
        except AudioValidationError as exc:
            return _error_result(str(exc), start)

        try:
            padded = pad_to_fixed_length(waveform)
            synthetic_probability = self._adapter.predict(padded)
        except Exception as exc:  # model inference failure -- don't hide it
            return _error_result(f"inference failed: {exc}", start)

        inference_ms = (time.perf_counter() - start) * 1000.0
        return {
            "synthetic_probability": synthetic_probability,
            "artifact_flags": flags,
            "model_name": MODEL_NAME,
            "inference_ms": inference_ms,
            "error": None,
        }

    async def predict_async(self, audio_bytes: bytes) -> DetectionResult:
        """
        Async prediction for the streaming backend. Runs the (synchronous,
        CPU-bound) model call in a thread so it doesn't block the event
        loop that's also running the WebSocket and the other two pipeline
        branches (NLP, biometrics) via asyncio.gather.
        """
        return await asyncio.to_thread(self.predict, audio_bytes)


def _error_result(message: str, start_time: float) -> DetectionResult:
    return {
        "synthetic_probability": None,
        "artifact_flags": [],
        "model_name": MODEL_NAME,
        "inference_ms": (time.perf_counter() - start_time) * 1000.0,
        "error": message,
    }
