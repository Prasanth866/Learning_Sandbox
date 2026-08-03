"""
Upgraded FastAPI + WebSocket + asyncio.Queue example.

Builds on the "queue feeds a background worker" pattern, but adds the
pieces you actually need for a coding-agent backend:

  1. Task cancellation (client can stop a running job)
  2. Streaming output from a real subprocess, line by line
  3. Timeouts on long-running work
  4. Graceful handling of client disconnect mid-task
  5. A small worker pool instead of one serial worker
  6. Pydantic validation of incoming messages
  7. `lifespan` instead of the deprecated `on_event`
  8. A bounded queue for backpressure

Run with: uvicorn agent_backend:app --reload
Test with a simple websocket client (e.g. `websocat ws://localhost:8000/ws/client1`)
and send: {"action": "start_task", "task_name": "list_files", "command": "ls -la"}
"""

import asyncio
import json
import shlex
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# 1. Data model for incoming client messages
# ---------------------------------------------------------------------------

class StartTaskMessage(BaseModel):
    action: str
    task_name: str = "unnamed_task"
    command: str = "echo hello"
    timeout_seconds: float = 15.0


# ---------------------------------------------------------------------------
# 2. Per-client job tracking, so we can cancel in-flight work
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
        return state

    async def _writer_loop(self, client_id: str, state: ClientState):
        try:
            while True:
                event = await state.outbox.get()
                try:
                    await state.websocket.send_text(json.dumps(event))
                    state.consecutive_send_failures = 0
                except Exception:
                    state.consecutive_send_failures += 1
                    if state.consecutive_send_failures >= MAX_SEND_FAILURES:
                        self.disconnect(client_id)
                        return
        except asyncio.CancelledError:
            pass

    def disconnect(self, client_id: str):
        state = self.clients.pop(client_id, None)
        if state is None:
            return
        if state.current_task and not state.current_task.done():
            state.current_task.cancel()
        if state.writer_task and not state.writer_task.done():
            state.writer_task.cancel()

    async def send_event(self, client_id: str, event: dict):
        """Enqueue an event for delivery. Never calls send_text directly —
        that's the writer task's job, and only its job."""
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
# 3. The actual "work": run a subprocess and stream its stdout line by line.
#    This is the shape you'll reuse later for "run agent-generated code" or
#    "stream LLM tokens" — replace the subprocess call with your real logic.
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
            proc.kill()
            await proc.wait()
        await manager.send_event(client_id, {
            "event": "TASK_CANCELLED",
            "task": msg.task_name,
        })
        raise

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
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
# 4. Worker pool: several workers pull from the same queue concurrently.
# ---------------------------------------------------------------------------

async def worker(worker_id: int):
    while True:
        client_id, msg = await job_queue.get()
        print(f"[worker {worker_id}] picked up '{msg.task_name}' for client {client_id}")
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
            job_queue.task_done()


# ---------------------------------------------------------------------------
# 4b. Reaper: periodically sweep for clients that have gone silent. This is
#     what catches "dead" connections — sockets that never raised
#     WebSocketDisconnect because the network just vanished (sleep, wifi
#     drop, phone backgrounded) rather than closing cleanly.
# ---------------------------------------------------------------------------

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
            if state.current_task and not state.current_task.done():
                state.current_task.cancel()  # don't leak a running job
            if state.writer_task and not state.writer_task.done():
                state.writer_task.cancel()
            try:
                await state.websocket.close(code=1001, reason="heartbeat timeout")
            except Exception:
                pass  # already gone; closing is best-effort
            manager.clients.pop(cid, None)


# ---------------------------------------------------------------------------
# 5. Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    workers = [asyncio.create_task(worker(i)) for i in range(WORKER_POOL_SIZE)]
    reaper_task = asyncio.create_task(reaper())
    yield
    reaper_task.cancel()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, reaper_task, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# 6. WebSocket endpoint: validates input, supports start + cancel
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
            except asyncio.TimeoutError:
                await manager.send_event(client_id, {"event": "PING"})
                continue

            state.last_seen = time.monotonic()

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
        manager.disconnect(client_id)