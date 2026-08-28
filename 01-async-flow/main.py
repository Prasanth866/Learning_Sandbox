import asyncio
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from config import HEARTBEAT_INTERVAL, WORKER_POOL_SIZE, logger
from manager import manager
from models import StartTaskMessage
from worker import job_queue, reaper, worker

@asynccontextmanager
async def lifespan(_app: FastAPI):
    workers = [asyncio.create_task(worker(i)) for i in range(WORKER_POOL_SIZE)]
    reaper_task = asyncio.create_task(reaper())
    logger.info("Server started. Background workers and reaper initialized.")
    yield
    reaper_task.cancel()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, reaper_task, return_exceptions=True)

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    state = await manager.connect(client_id, websocket)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=HEARTBEAT_INTERVAL
                )
                state.last_seen = time.monotonic()
            except asyncio.TimeoutError:
                await manager.send_event(client_id, {"event": "PING"})
                continue

            try:
                payload = json.loads(raw)
                msg = StartTaskMessage(**payload)
            except (json.JSONDecodeError, ValidationError) as e:
                await manager.send_event(client_id, {
                    "event": "ERROR",
                    "error": f"Invalid message: {e}",
                })
                continue

            if msg.action == "pong":
                continue

            if msg.action == "start_task":
                try:
                    job_queue.put_nowait((client_id, msg))
                    await manager.send_event(client_id, {
                        "event": "TASK_QUEUED",
                        "task": msg.task_name,
                        "queue_position": job_queue.qsize(),
                    })
                except asyncio.QueueFull:
                    await manager.send_event(client_id, {
                        "event": "ERROR",
                        "error": "Server busy, try again shortly.",
                    })

            elif msg.action == "cancel_task":
                if state.current_task and not state.current_task.done():
                    state.current_task.cancel()
                else:
                    await manager.send_event(client_id, {
                        "event": "ERROR",
                        "error": "No task currently running.",
                    })

    except WebSocketDisconnect:
        manager.disconnect(client_id, reason="clean_disconnect")
