# FastAPI + WebSockets + `asyncio.Queue` — Coding Agent Backend (Learning Project)

A self-contained mini-project for learning the **asyncio + WebSocket** patterns needed to build a coding-agent backend. A client sends a command over a WebSocket, a pool of background workers executes it, and the output is streamed back to the client in real time.

> **Goal:** Learn the concurrency, streaming, and lifecycle patterns used by modern AI coding agents.

---

## Overview

This explores the kinds of problems you'll encounter once a WebSocket server starts handling **real, long-running work**, including:

- Task cancellation
- Execution timeouts
- Dead WebSocket connections
- Client/server state desynchronization
- Concurrent writes to the same socket
- Backpressure and queue management

---

## Security Warning

> **Do NOT expose this project to untrusted users or the public internet.**

The endpoint:

```text
/ws/{client_id}
```

executes **whatever command** the connected client sends.

This is intentional for learning the subprocess-streaming pattern, but **must not** be deployed without sandboxing, authentication, and command restrictions.

See **Known Limitations** below.

---

# What You'll Learn

## Producer / Consumer Architecture

- WebSocket handler only accepts requests.
- Long-running work is pushed into an `asyncio.Queue`.
- Background workers process jobs independently.
- The WebSocket handler never blocks while work executes.

---

## Worker Pool

Instead of a single worker:

- Multiple background workers run concurrently.
- Different clients can execute commands simultaneously.
- Prevents every request from waiting behind one long-running job.

---

## Task Cancellation & Cleanup

Supports cancelling a running job using:

```python
asyncio.Task.cancel()
```

Proper cleanup includes:

- handling `CancelledError`
- terminating the subprocess
- removing stale task references
- preventing phantom running-task state

---

## Execution Timeouts

Each job is wrapped with:

```python
asyncio.wait_for(...)
```

Benefits:

- prevents workers from hanging forever
- automatically cancels long-running commands
- keeps the worker pool healthy

---

## Safe Subprocess Cancellation

Handles race conditions such as:

- process exits naturally
- cancellation occurs simultaneously
- timeout triggers during shutdown

Gracefully avoids exceptions like:

```text
ProcessLookupError
```

---

## Streaming Subprocess Output

Uses:

```python
asyncio.create_subprocess_exec()
```

and reads:

```python
stdout.readline()
```

This closely matches how LLM token streaming works.

```
LLM
 ↓
token
 ↓
WebSocket
 ↓
Client
```

---

## Dead Client Detection

Implements an application-level heartbeat.

Features:

- PING/PONG messages
- last_seen timestamp
- periodic cleanup ("reaper")

Detects failures such as:

- Wi-Fi disconnects
- browser crashes
- backgrounded mobile apps
- silent socket drops

which often **do not** trigger `WebSocketDisconnect`.

---

## Single Writer Per Connection

Each client owns:

```
WebSocket
      ↑
 Writer Task
      ↑
 Outbox Queue
```

Only one coroutine is allowed to call:

```python
send_text(...)
```

This prevents:

- concurrent frame corruption
- runtime send errors
- interleaved messages

---

## Input Validation

Uses **Pydantic** models instead of:

```python
data.get(...)
```

Benefits:

- schema validation
- automatic type checking
- cleaner code
- safer parsing

---

## Backpressure

Every queue is bounded.

Examples:

- job queue
- client outbox queue

Prevents:

- unlimited memory growth
- slow clients exhausting RAM
- request floods

---

## Modern FastAPI Lifecycle

Uses:

```python
lifespan()
```

instead of the deprecated:

```python
@app.on_event("startup")
```

---

# Project Structure

```text
.
├── agent_backend.py      # Complete implementation
├── requirements.txt
└── README.md
```

---

# Installation

```bash
python3 -m venv venv

source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

---

# Running

```bash
uvicorn agent_backend:app --reload
```

Server endpoint:

```text
ws://localhost:8000/ws/{client_id}
```

where `client_id` is any identifier you choose.

---

# Trying It Out

Connect using **websocat** (or any WebSocket client):

```bash
websocat ws://localhost:8000/ws/client1
```

---

## Start a Task

```json
{
  "action": "start_task",
  "task_name": "list_files",
  "command": "ls -la",
  "timeout_seconds": 10
}
```

---

## Cancel a Task

```json
{
  "action": "cancel_task"
}
```

---

## Heartbeat

```json
{
  "action": "pong"
}
```

---

# Event Flow

Typical execution:

```
TASK_QUEUED
      ↓
TASK_STARTED
      ↓
TASK_OUTPUT
      ↓
TASK_OUTPUT
      ↓
TASK_OUTPUT
      ↓
TASK_FINISHED
```

Possible terminal events:

- `TASK_FINISHED`
- `TASK_CANCELLED`
- `TASK_FAILED`

---

# Heartbeat Behavior

Every **20 seconds** of inactivity:

```json
{
  "event": "PING"
}
```

Client should respond with:

```json
{
  "action": "pong"
}
```

Any valid command also refreshes the heartbeat timer.

If no activity occurs for **45+ seconds**, the reaper closes the dead connection.

---

# Project Evolution

The project intentionally evolved through several realistic iterations.

## v1 — Minimal Queue Worker

Features:

- one worker
- `@app.on_event("startup")`
- no validation
- no cancellation

Purpose:

- understand the basic producer/consumer pattern

---

## v2 — Agent-Oriented Backend

Added:

- worker pool
- subprocess streaming
- task cancellation
- execution timeouts
- Pydantic models
- lifespan startup

---

## v3 — Dead Client Handling

Added:

- heartbeat (PING/PONG)
- background reaper

Reason:

`WebSocketDisconnect` only detects clean disconnects.

---

## v4 — Single Writer Pattern

Introduced:

- per-client outbox queue
- dedicated writer task

Fixed:

- concurrent `send_text()` race conditions

---

## v5 — State & Cancellation Hardening

Fixed several subtle issues:

- PONG messages now refresh heartbeat timestamps
- prevented `ProcessLookupError` during race-condition cancellation
- explicit worker cleanup
- stale task reference removal

---

> Reading **why** each version exists is more valuable than reading the final implementation. Every iteration fixes a real class of production bugs commonly encountered in asynchronous systems.

---

# Known Limitations

## 1. Arbitrary Command Execution

Commands are executed directly via:

```python
asyncio.create_subprocess_exec(...)
```

There is:

- no sandbox
- no allowlist
- no resource limits

A production system should use:

- command allowlists
- isolated containers or VMs
- CPU limits
- memory limits
- filesystem isolation

---

## 2. No Authentication

Any client can connect using any:

```text
client_id
```

No authentication or authorization exists.

---

## 3. In-Memory State

Everything lives inside one process:

- connected clients
- queues
- worker state

Consequences:

- server restart loses all state
- horizontal scaling isn't supported

A production deployment would move shared state into something like:

- Redis
- PostgreSQL
- distributed queues

---

## 4. Queued Job Cancellation

Cancellation only affects the **currently running task**.

If a job is still waiting inside:

```python
job_queue
```

it cannot be cancelled until a worker dequeues it and marks it as active.

---

## 5. `create_subprocess_exec()` by Design

The project intentionally uses:

```python
asyncio.create_subprocess_exec(...)
```

instead of:

```python
asyncio.create_subprocess_shell(...)
```

This avoids invoking a shell, meaning shell metacharacters are treated as plain text.

As a result:

- No pipes (`|`)
- No redirection (`>`)
- No shell variable expansion (`$HOME`)

do **not** work.

This is a deliberate security tradeoff to reduce command-injection risks rather than an implementation limitation.

