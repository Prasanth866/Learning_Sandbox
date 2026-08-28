import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("async_flow_backend")

HEARTBEAT_INTERVAL = 20.0
CLIENT_TIMEOUT = 45.0
REAPER_SWEEP_INTERVAL = 10.0
MAX_SEND_FAILURES = 3
WORKER_POOL_SIZE = 3
JOB_QUEUE_MAXSIZE = 50
OUTBOX_MAXSIZE = 100
