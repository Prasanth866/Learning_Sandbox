"""
Upgraded FastAPI + WebSocket + asyncio.Queue example.

Builds on the "queue feeds a background worker" pattern, with production-ready
lifecycle management, task cancellation, streaming, and heartbeats.

Run with: uvicorn agent_backend:app --reload
Test with: websocat ws://localhost:8000/ws/client1
"""

import asyncio
import json
import logging
import shlex
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("asyncflow_backend")

# ---------------------------------------------------------------------------
# 1. Data model for incoming client messages
# ---------------------------------------------------------------------------

class StartTaskMessage(BaseModel):
    action: str
    task_name: str = "unnamed_task"
    command: str = "echo hello"
    timeout_seconds: float = 15.0


# ---------------------------------------------------------------------------
# 2. Per-client job tracking and Connection Management
# ---------------------------------------------------------------------------

@dataclass
class ClientState:
    websocket: WebSocket
    current_task: Optional[asyncio.Task] = None
    outbox: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    writer_task: Optional[asyncio.Task] = None
    last_seen: float = field(default_factory=time.monotonic)
    consecutive_send_failures: int = 0


HEARTBEAT_INTERVAL = 20.0
CLIENT_TIMEOUT = 45.0
REAPER_SWEEP_INTERVAL = 10.0
MAX_SEND_FAILURES = 3


class ConnectionManager:
    def __init__(self):
        self.clients: Dict[str, ClientState] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> ClientState:
        await websocket.accept()
        state = ClientState(websocket=websocket)
        self.clients[client_id] = state
        state.writer_task = asyncio.create_task(self._writer_loop(client_id, state))
        logger.info(f"[connect] Client '{client_id}' connected.")
        return state

    async def _writer_loop(self, client_id: str, state: ClientState):
        try:
            while True:
                event = await state.outbox.get()
                try:
                    await state.websocket.send_text(json.dumps(event))
                    state.consecutive_send_failures = 0
                except Exception as e:
                    state.consecutive_send_failures += 1
                    logger.warning(
                        f"[writer] Failed sending to '{client_id}' "
                        f"({state.consecutive_send_failures}/{MAX_SEND_FAILURES}): {e}"
                    )
                    if state.consecutive_send_failures >= MAX_SEND_FAILURES:
                        logger.error(f"[writer] Dropping client '{client_id}' after consecutive write failures.")
                        self.disconnect(client_id, reason="max_send_failures")
                        return
        except asyncio.CancelledError:
            pass

    def disconnect(self, client_id: str, reason: str = "client_disconnected"):
        state = self.clients.pop(client_id, None)
        if state is None:
            return

        logger.info(f"[disconnect] Client '{client_id}' disconnected (Reason: {reason}).")

        if state.current_task and not state.current_task.done():
            state.current_task.cancel()
        if state.writer_task and not state.writer_task.done():
            state.writer_task.cancel()

    async def send_event(self, client_id: str, event: dict):
        """Enqueue an event for delivery via the client's outbox writer loop."""
        state = self.clients.get(client_id)
        if state is None:
            return
        try:
            state.outbox.put_nowait(event)
        except asyncio.QueueFull:
            try:
                state.outbox.get_nowait()
                state.outbox.put_nowait(event)
            except asyncio.QueueEmpty:
                pass


manager = ConnectionManager()
job_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
WORKER_POOL_SIZE = 3


# ---------------------------------------------------------------------------
# 3. Subprocess execution with live output streaming
# ---------------------------------------------------------------------------

async def run_streaming_job(client_id: str, msg: StartTaskMessage):
    await manager.send_event(client_id, {
        "event": "TASK_STARTED",
        "task": msg.task_name,
    })

    proc = None
    try:
        try:
            args = shlex.split(msg.command)
        except ValueError as e:
            raise RuntimeError(f"Could not parse command: {e}") from e

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def stream_output(proc: asyncio.subprocess.Process):
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="replace").rstrip()
                await manager.send_event(client_id, {
                    "event": "TASK_OUTPUT",
                    "task": msg.task_name,
                    "line": line,
                })

        await asyncio.wait_for(stream_output(proc), timeout=msg.timeout_seconds)
        return_code = await proc.wait()

        await manager.send_event(client_id, {
            "event": "TASK_FINISHED",
            "task": msg.task_name,
            "status": "SUCCESS" if return_code == 0 else "ERROR",
            "return_code": return_code,
        })

    except asyncio.CancelledError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        await manager.send_event(client_id, {
            "event": "TASK_CANCELLED",
            "task": msg.task_name,
        })
        raise

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        await manager.send_event(client_id, {
            "event": "TASK_FAILED",
            "task": msg.task_name,
            "error": f"Timed out after {msg.timeout_seconds}s",
        })

    except Exception as e:
        await manager.send_event(client_id, {
            "event": "TASK_FAILED",
            "task": msg.task_name,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# 4. Worker pool & Reaper background tasks
# ---------------------------------------------------------------------------

async def worker(worker_id: int):
    while True:
        client_id, msg = await job_queue.get()
        logger.info(f"[worker {worker_id}] picked up '{msg.task_name}' for client '{client_id}'")
        try:
            state = manager.clients.get(client_id)
            if state is None:
                continue

            task = asyncio.create_task(run_streaming_job(client_id, msg))
            state.current_task = task
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if state.current_task == task:
                    state.current_task = None
        finally:
            job_queue.task_done()


async def reaper():
    while True:
        await asyncio.sleep(REAPER_SWEEP_INTERVAL)
        now = time.monotonic()
        stale_ids = [
            cid for cid, state in manager.clients.items()
            if now - state.last_seen > CLIENT_TIMEOUT
        ]
        for cid in stale_ids:
            state = manager.clients.get(cid)
            if state is None:
                continue

            logger.warning(f"[reaper] Client '{cid}' timed out (inactive for {now - state.last_seen:.1f}s). Closing connection.")

            if state.current_task and not state.current_task.done():
                state.current_task.cancel()
            if state.writer_task and not state.writer_task.done():
                state.writer_task.cancel()

            try:
                await state.websocket.close(code=1001, reason="heartbeat timeout")
            except Exception as e:
                logger.debug(f"[reaper] Socket close failed for '{cid}': {e}")

            manager.disconnect(cid, reason="heartbeat_timeout")


# ---------------------------------------------------------------------------
# 5. Lifespan Context Manager
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. WebSocket Endpoint
# ---------------------------------------------------------------------------

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