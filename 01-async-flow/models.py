import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket
from pydantic import BaseModel
from config import OUTBOX_MAXSIZE

class StartTaskMessage(BaseModel):
    action: str
    task_name: str = "unnamed_task"
    command: str = "echo hello"
    timeout_seconds: float = 15.0

@dataclass
class ClientState:
    websocket: WebSocket
    current_task: Optional[asyncio.Task] = None
    outbox: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=OUTBOX_MAXSIZE))
    writer_task: Optional[asyncio.Task] = None
    last_seen: float = field(default_factory=time.monotonic)
    consecutive_send_failures: int = 0
