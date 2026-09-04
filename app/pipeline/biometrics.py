
from pathlib import Path
import tempfile

from speechbrain.inference.speaker import SpeakerRecognition


# Load the ECAPA-TDNN speaker verification model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Location of the enrolled/reference voice
TRUSTED_VOICE = Path("data") / "enrolled" / "trusted_voice.wav"

# Similarity threshold above which we consider it the same speaker
# Tune this after running real vs impostor tests
MATCH_THRESHOLD = 0.35


async def verify_speaker_voiceprint(audio_bytes: bytes) -> dict:
    """
    Compare incoming audio with the enrolled trusted voice.

    Returns:
        dict: {
            "similarity_score": float, similarity between the two voices,
            "is_match": bool, whether score clears MATCH_THRESHOLD
        }
    """

    # Save the incoming audio bytes temporarily as a WAV file
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
        dir="."
    ) as temp_audio:

        temp_audio.write(audio_bytes)
        incoming_audio = Path(temp_audio.name).name

    try:
        # Compare incoming voice with trusted enrolled voice
        score, prediction = verification.verify_files(
            str(TRUSTED_VOICE),
            incoming_audio
        )

        similarity_score = float(score.item())
        is_match = similarity_score >= MATCH_THRESHOLD

        print("Similarity score:", similarity_score)
        print("Same speaker (model prediction):", prediction.item())
        print("Same speaker (threshold decision):", is_match)

        return {
            "similarity_score": similarity_score,
            "is_match": is_match,
        }

    finally:
        # Delete the temporary audio file
        Path(incoming_audio).unlink(missing_ok=True)