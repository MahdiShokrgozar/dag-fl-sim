"""
Wall-time accumulators for the simulator's own cost (tool paper, scalability study E1).

They only read the clock; they never touch model state, the event queue or any random
stream, so a run's results are unchanged. main.py writes the totals, with peak memory,
to run_resources.json at the end of a run.
"""
import time
from collections import defaultdict
from contextlib import contextmanager

TOTALS = defaultdict(float)
COUNTS = defaultdict(int)


@contextmanager
def timed(key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        TOTALS[key] += time.perf_counter() - t0
        COUNTS[key] += 1
