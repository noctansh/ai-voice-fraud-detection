import asyncio
import websockets
import json

async def simulate_audio_stream():
    uri = "ws://localhost:8000/ws/stream"
    print("Connecting to Fraud Shield WebSocket server...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected successfully! Starting audio buffer streaming...")

        dummy_chunk = b"\x00" * 3200

        async def listen_for_scores():
            try:
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print(f"--> Live Telemetry: Risk = {data['risk_score']}% | Synthetic = {data['synthetic_score']} | Urgency = {data['urgency_score']} | Alert = {data['alert']}")
            except websockets.exceptions.ConnectionClosed:
                pass

        listener = asyncio.create_task(listen_for_scores())

        for _ in range(25):
            await websocket.send(dummy_chunk)
            await asyncio.sleep(0.1)

        await asyncio.sleep(2)
        listener.cancel()
        print("Stream simulation complete.")

if __name__ == "__main__":
    asyncio.run(simulate_audio_stream())