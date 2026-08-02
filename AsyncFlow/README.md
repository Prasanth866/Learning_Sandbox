# FastAPI + WebSockets + asyncio.Queue — Coding Agent Backend (Learning Project)

A self-contained mini-project for learning the asyncio + WebSocket patterns
needed to build a coding-agent backend: a client sends a command over a
WebSocket, a pool of background workers runs it, and streaming output flows
back to the client in real time.

This isn't a toy "hello world" — it deliberately works through the failure
modes you actually hit once a WebSocket server does real, long-running work:
cancellation, timeouts, dead connections, and concurrent writes to the same
socket.

> **Not safe for untrusted network exposure.** The `/ws/{client_id}`
> endpoint executes whatever command a connected client sends it. This is
> intentional for learning the subprocess-streaming pattern, but do **not**
> deploy this as-is anywhere reachable by the public internet or by users
> you don't trust. See [Known limitations](#known-limitations) below.

## What this demonstrates

- **Producer/consumer decoupling** — the WebSocket handler only enqueues
  work; a separate pool of workers processes it. The handler never blocks
  on long-running jobs.
- **A worker pool**, not a single worker — several jobs (from different
  clients) can run concurrently instead of queuing serially behind one
  worker.
- **Task cancellation** — a client can stop a running job mid-flight
  (`asyncio.Task.cancel()` + proper `CancelledError` handling), and the
  underlying subprocess is killed along with it.
- **Timeouts** — every job is bounded with `asyncio.wait_for`, so a hung
  command can't tie up a worker forever.
- **Streaming subprocess output** — `asyncio.create_subprocess_exec` +
  reading `stdout` line-by-line, the same shape you'd use to stream tokens
  from an LLM call instead.
- **Dead-client detection** — an application-level heartbeat (ping on idle,
  a periodic "reaper" sweep) catches connections that go silent without a
  clean disconnect (sleep, dropped wifi, backgrounded app) — something
  `WebSocketDisconnect` alone will never catch.
- **Single-writer-per-connection** — every outgoing message is routed
  through a per-client `outbox` queue, with exactly one dedicated task
  responsible for calling `send_text()`. This is what prevents concurrent
  sends on the same socket, which can otherwise corrupt frames or crash the
  connection.
- **Input validation** with Pydantic instead of raw `dict.get()`.
- **Backpressure** — bounded queues everywhere (job queue, per-client
  outbox) so a flood of requests or a slow client can't grow memory
  unbounded.
- **Modern FastAPI lifecycle** — `lifespan` context manager instead of the
  deprecated `@app.on_event("startup")`.

## Project structure

```
.
├── agent_backend.py   # the whole implementation
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
uvicorn agent_backend:app --reload
```

The server starts on `ws://localhost:8000/ws/{client_id}`, where
`client_id` is any string you choose to identify the connection.

## Trying it out

Using [`websocat`](https://github.com/vi/websocat) (or any WebSocket
client — a browser console works fine too):

```bash
websocat ws://localhost:8000/ws/client1
```

Then send JSON messages:

**Start a task:**
```json
{"action": "start_task", "task_name": "list_files", "command": "ls -la", "timeout_seconds": 10}
```

**Cancel the running task:**
```json
{"action": "cancel_task"}
```

You should see a stream of events back: `TASK_QUEUED` → `TASK_STARTED` →
repeated `TASK_OUTPUT` lines → `TASK_FINISHED` (or `TASK_CANCELLED` /
`TASK_FAILED`). Roughly every 20 seconds of silence you'll also see a
`PING` — try replying with `{"action": "pong"}`, or just leave the
connection idle for 45+ seconds and watch the server force-close it.

## How it evolved

This project started as a minimal queue-fed-worker example and was
hardened in stages — each stage fixing a specific, realistic bug rather
than adding a feature for its own sake:

1. **v1 — naive**: one worker, `on_event("startup")`, no cancellation, no
   validation. Good for understanding the basic producer/consumer shape,
   bad for anything real.
2. **v2 — agent-shaped**: added cancellation, subprocess streaming,
   timeouts, a worker pool, Pydantic validation, `lifespan`.
3. **v3 — dead-client handling**: added an application-level heartbeat
   (ping/pong) and a background reaper task, since `WebSocketDisconnect`
   only fires on a *clean* close and silently-dropped connections would
   otherwise leak tasks and subprocesses forever.
4. **v4 — race-condition fix**: discovered that multiple coroutines
   (workers, the heartbeat, the reaper) could call `send_text()` on the
   same socket concurrently. Fixed by routing all outgoing messages through
   a bounded `outbox` queue with a single dedicated writer task per
   connection — the general pattern for serializing writes to any shared,
   stateful resource.

If you're using this to learn rather than just to run, reading the diffs
between these stages is arguably more valuable than reading the final file
— each one exists because of a specific bug class, not because more code is
better.

## Known limitations

- **Arbitrary command execution.** `command` is executed directly via
  `asyncio.create_subprocess_exec` with no sandboxing, allowlisting, or
  resource limits. This is fine for a local learning sandbox; it is not
  fine for anything with real users. A production version would need at
  minimum a strict command allowlist, or better, execution inside an
  isolated container/VM with CPU, memory, and filesystem limits.
- **No authentication.** Any client can connect as any `client_id`.
- **In-memory only.** All state (`ConnectionManager.clients`, both queues)
  lives in a single process. Restarting the server drops all connections
  and queued jobs; running multiple instances behind a load balancer would
  require moving state to something shared (e.g. Redis) instead.
- **`create_subprocess_exec`, not `create_subprocess_shell`**, is used on
  purpose — it never invokes a shell, so shell metacharacters in `command`
  are inert. This means pipes/redirects/env expansion in `command` won't
  work as a shell would interpret them; that's a deliberate tradeoff to
  avoid reopening command injection, not an oversight.

## What's next

The natural extension of this project is swapping the subprocess-streaming
job (`run_streaming_job`) for a real LLM call that streams tokens via an
async generator instead of subprocess `stdout` lines — same
`async for chunk in source` shape, different source.
