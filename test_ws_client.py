# test_ws_client.py
import asyncio
import websockets

async def test_connection():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Successfully connected to {uri}")
            # Optional: Send a test message
            await websocket.send("Hello from test client!")
            response = await websocket.recv()
            print(f"Received: {response}")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())