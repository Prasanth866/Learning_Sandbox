import asyncio
import shlex
from manager import manager
from models import StartTaskMessage

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
