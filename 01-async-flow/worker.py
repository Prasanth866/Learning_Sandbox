import asyncio
import time
from config import CLIENT_TIMEOUT, JOB_QUEUE_MAXSIZE, REAPER_SWEEP_INTERVAL, logger
from manager import manager
from runner import run_streaming_job

job_queue: asyncio.Queue = asyncio.Queue(maxsize=JOB_QUEUE_MAXSIZE)

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
