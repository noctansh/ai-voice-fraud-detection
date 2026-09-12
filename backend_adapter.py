"""
Drop-in replacement for the current stub in Ansh's repo:

    # app/pipeline/deepfake.py (existing)
    async def analyze_acoustic_artifacts(audio_bytes: bytes) -> float:
        # TODO (Vansh): Run ONNX/Wav2Vec2 model
        return 0.15

To wire in the real detector, Ansh only needs to change app/pipeline/deepfake.py
to:

    from models.deepfake.backend_adapter import analyze_acoustic_artifacts

No changes needed anywhere else in his backend -- same name, same signature,
same return type (a bare float), so app/main.py's asyncio.gather(...) call
keeps working unmodified.

Note: this collapses the rich DetectionResult dict down to a single float,
which loses the artifact_flags/model_name/error detail. That's fine for the
live pipeline (Ansh only needs the number), but means load/inference errors
become a fallback score rather than a visible error. We fail toward "flag as
suspicious" (1.0) rather than "flag as safe" (0.0) on error, since silently
returning a low score on failure would be worse for a fraud-detection tool.
"""
from __future__ import annotations

from .detector import AcousticDeepfakeDetector

_detector: AcousticDeepfakeDetector = None  # lazy singleton -- one model load, reused


def _get_detector() -> AcousticDeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = AcousticDeepfakeDetector()
    return _detector


async def analyze_acoustic_artifacts(audio_bytes: bytes) -> float:
    result = await _get_detector().predict_async(audio_bytes)
    if result["error"] is not None or result["synthetic_probability"] is None:
        # Fail loud in logs, fail safe (high suspicion) in the score.
        print(f"[deepfake] detector error, flagging as suspicious: {result['error']}")
        return 1.0
    return result["synthetic_probability"]
