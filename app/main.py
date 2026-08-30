import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.models.schemas import AnalysisResult
from app.scoring import calculate_risk_index
from app.pipeline.deepfake import analyze_acoustic_artifacts
from app.pipeline.nlp_intent import analyze_intent_and_transcribe
from app.pipeline.biometrics import verify_speaker_voiceprint

app = FastAPI(title="AI Voice Clone Fraud Detection Backend")

@app.get("/")
def home():
    return {"message": "Voice Fraud Detection Backend is running!"}

@app.websocket("/ws/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    BUFFER_THRESHOLD = 64000  # 2 seconds of 16kHz mono PCM

    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)

            if len(buffer) >= BUFFER_THRESHOLD:
                active_window = bytes(buffer[:BUFFER_THRESHOLD])
                buffer = buffer[int(BUFFER_THRESHOLD * 0.75):]

                synthetic_score, (transcript, urgency_score), mismatch_score = await asyncio.gather(
                    analyze_acoustic_artifacts(active_window),
                    analyze_intent_and_transcribe(active_window),
                    verify_speaker_voiceprint(active_window),
                )

                risk_index = calculate_risk_index(synthetic_score, mismatch_score, urgency_score)

                response = AnalysisResult(
                    synthetic_score=synthetic_score,
                    speaker_mismatch=mismatch_score,
                    urgency_score=urgency_score,
                    transcript=transcript,
                    risk_score=risk_index,
                    alert=(risk_index > 75.0)
                )

                await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        print("Client disconnected.")