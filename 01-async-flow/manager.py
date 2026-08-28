import asyncio
import json
from typing import Dict
from fastapi import WebSocket
from config import MAX_SEND_FAILURES, logger
from models import ClientState

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
