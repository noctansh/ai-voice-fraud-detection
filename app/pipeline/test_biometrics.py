import asyncio
from pathlib import Path

from biometrics import verify_speaker_voiceprint


async def main():
    # Genuine case: should match the trusted voice
    genuine_audio = Path("myaudio1.wav").read_bytes()
    genuine_result = await verify_speaker_voiceprint(genuine_audio)
    print("Genuine case result:", genuine_result)

    print("-" * 40)

    # Impostor case: different speaker, should NOT match
    impostor_audio = Path("myaudio2.wav").read_bytes()
    impostor_result = await verify_speaker_voiceprint(impostor_audio)
    print("Impostor case result:", impostor_result)


asyncio.run(main())
 