from collections import deque
from threading import Lock
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAX_RESULTS = 3

_results = deque(maxlen=MAX_RESULTS)
_lock = Lock()

logger.info(f"Order results storage initialized: max_results={MAX_RESULTS}")

def add_result(result: dict):
    with _lock:
        order_id = result.get('order_id', 'unknown')
        status = result.get('status', 'unknown')
        logger.info(f"[RESULTS] Adding result: order_id={order_id}, status={status}")
        logger.debug(f"[RESULTS] Result details: {result}")
        _results.appendleft(result)
        logger.debug(f"[RESULTS] Current result count: {len(_results)}/{MAX_RESULTS}")

def get_results():
    with _lock:
        result_list = list(_results)
        logger.debug(f"[RESULTS] Retrieved {len(result_list)} results")
        return result_list
