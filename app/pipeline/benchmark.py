import asyncio
from pathlib import Path

from biometrics import verify_speaker_voiceprint

GENUINE_DIR = Path("data")/"enrolled" / "test_samples" / "genuine"
IMPOSTOR_DIR = Path("data")/"enrolled" / "test_samples" / "imposter"


async def run_benchmark():
    total = 0
    correct = 0
    false_accepts = 0
    false_rejects = 0

    print("=== Genuine samples (should MATCH) ===")
    for audio_file in GENUINE_DIR.glob("*.wav"):
        result = await verify_speaker_voiceprint(audio_file.read_bytes())
        total += 1
        status = "✅ correct" if result["is_match"] else "❌ FALSE REJECT"
        if not result["is_match"]:
            false_rejects += 1
        else:
            correct += 1
        print(f"{audio_file.name}: score={result['similarity_score']:.3f} | {status}")

    print("\n=== Impostor samples (should NOT match) ===")
    for audio_file in IMPOSTOR_DIR.glob("*.wav"):
        result = await verify_speaker_voiceprint(audio_file.read_bytes())
        total += 1
        status = "✅ correct" if not result["is_match"] else "❌ FALSE ACCEPT"
        if result["is_match"]:
            false_accepts += 1
        else:
            correct += 1
        print(f"{audio_file.name}: score={result['similarity_score']:.3f} | {status}")

    print("\n=== Summary ===")
    print(f"Total samples: {total}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print(f"False Accept Rate: {false_accepts}/{total}")
    print(f"False Reject Rate: {false_rejects}/{total}")


asyncio.run(run_benchmark())
