# async-flow

Concise notes on the concurrency patterns, lifecycle handling, and WebSocket streaming architecture in `async-flow`.

---

## 1. What async-flow Is

An in-memory backend built with **FastAPI**, **WebSockets**, and **`asyncio`** that simulates how AI coding agents run terminal commands, stream outputs in real time, and handle task lifecycles safely.

```
[ Client ] ──WebSocket──► [ Endpoint ] ──enqueue──► [ asyncio.Queue ]
    ▲                                                      │
    │ (outbox stream)                                      ▼
[ Writer Task ] ◄──events── [ Worker Pool ] ◄──run── [ Subprocess ]
```

---

## 2. Core Architecture & Components

| Component | File | Responsibility |
|---|---|---|
| **`ConnectionManager` & `ClientState`** | `manager.py`, `models.py` | Tracks active clients, manages per-client outbox queues, and isolates outbound socket writes. |
| **`job_queue` (`asyncio.Queue`)** | `worker.py` | Bounded queue (capacity 50) holding `(client_id, StartTaskMessage)` tuples. |
| **Worker Pool (`worker`)** | `worker.py` | 3 long-running async worker tasks consuming jobs off `job_queue`. |
| **Streaming Runner (`run_streaming_job`)** | `runner.py` | Spawns subprocesses, streams lines from `stdout`, handles timeouts and cancellations. |
| **Reaper (`reaper`)** | `worker.py` | Background task sweeping every 10s to drop zombie connections silent for >45s. |
| **Lifespan Manager (`lifespan`)** | `main.py` | Initializes and cleans up worker tasks and the reaper on server startup/shutdown. |
| **WebSocket Handler (`/ws/{client_id}`)** | `main.py` | Parses client JSON messages, manages incoming heartbeat intervals, and dispatches actions. |
| **Configuration & Logging** | `config.py` | Application constants, timeouts, buffer sizes, and logger setup. |

---

## 3. Key Concurrency & Reliability Patterns

### 1. Producer–Consumer Separation
- The WebSocket handler never runs jobs directly; it validates messages with Pydantic and pushes to `job_queue`.
- Keeps the socket event loop responsive even under heavy execution loads.

### 2. Single Writer per Socket (Outbox Pattern)
- Direct calls to `websocket.send_text()` from multiple coroutines (worker output, ping, status updates) cause frame corruption and race conditions.
- **Solution:** Each client has a dedicated `outbox` queue and a single `_writer_loop` coroutine. All events route through `manager.send_event()`.

### 3. Real-Time Non-blocking Subprocess Streaming
- Uses `asyncio.create_subprocess_exec(*args, stdout=PIPE, stderr=STDOUT)` (merging stderr into stdout).
- Streams stdout line-by-line using `async for raw_line in proc.stdout:` and emits `TASK_OUTPUT` events immediately.

### 4. Safe Task Cancellation & Subprocess Teardown
- Client can cancel active tasks (`cancel_task` action).
- Caught in `asyncio.CancelledError`: explicitly kills running subprocess (`proc.kill()`) and handles `ProcessLookupError` if process already exited.
- Emits `TASK_CANCELLED` and clears `state.current_task`.

### 5. Execution Timeouts
- Subprocess streaming is wrapped with `asyncio.wait_for(stream_output(proc), timeout=msg.timeout_seconds)`.
- If timeout hits, terminates the subprocess and emits `TASK_FAILED`.

### 6. Application-Level Heartbeat & Reaper
- TCP socket disconnects aren't always detected immediately by `WebSocketDisconnect` (e.g. WiFi drop, hard kill).
- Server sends `PING` if client is idle for `HEARTBEAT_INTERVAL` (20s).
- Client sends `{"action": "pong"}` to update `last_seen`.
- `reaper()` sweeps every 10s and purges clients where `time.monotonic() - last_seen > 45s`.

### 7. Bounded Queues & Backpressure
- `job_queue` has `maxsize=50` (rejects with error if full).
- `outbox` has `maxsize=100` (drops oldest frame if full to prevent slow-reader memory leaks).

---

## 4. Protocol & Message Flow

### Client Actions (Inbound)

```json
// Start a command
{ "action": "start_task", "task_name": "build", "command": "ls -la", "timeout_seconds": 15 }

// Cancel running task
{ "action": "cancel_task" }

// Heartbeat response
{ "action": "pong" }
```

### Server Events (Outbound)

```
TASK_QUEUED ──► TASK_STARTED ──► TASK_OUTPUT (repeated) ──► TASK_FINISHED
                                                        ├──► TASK_CANCELLED
                                                        └──► TASK_FAILED
```

- System events: `PING`, `ERROR`

---

## 5. Running & Testing

```bash
# Run server
uvicorn main:app --reload

# Connect via websocat
websocat ws://localhost:8000/ws/client1
```

---

## 6. Implementation Notes & Tradeoffs

- **No Shell (`create_subprocess_exec`)**: Uses `shlex.split()` without a shell wrapper. Pipes (`|`), redirections (`>`), and env substitutions (`$VAR`) are not supported by design to prevent trivial shell injections.
- **In-Memory Only**: Client tracking and job queues live in process memory. Restarting the server resets all state (no Redis/DB persistence).
- **Queue vs Running Task Cancellation**: `cancel_task` only cancels the *currently executing* task on a worker. Tasks waiting in `job_queue` run until picked up unless handled by a deferred cancellation mechanism.
