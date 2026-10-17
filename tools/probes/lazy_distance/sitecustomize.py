"""
Lazy-distance probe (W4, thesis-hub/docs/decisions.md 2026-09-26).

Records, for every deep inspection, the lazy-gate distance a validator
computed (farabi_consensus logs it at DEBUG as [DIAG-DISTANCE]) together with
the proposer, the base size and whether the validator had its adaptive band
yet. Output: <run_dir>/lazy_distance.csv at process exit. The question: does
a lazy node that republishes a multi-parent Farabi-Prox average land at a
non-zero distance from the validator's median base, inside the honest range?

Put on PYTHONPATH with the repo root; no simulator code is changed.
"""

import atexit
import csv
import os
import re
import sys

_ROWS = []
_PAT = re.compile(
    r"proposer=(\S+) raw_distance=(\S+) lazy_check_distance=(\S+) base_n=(\d+) "
    r"aggregation_refs_n=(\d+) parents_n=(\d+)"
)


def _install():
    try:
        from src.strategies.consensus import farabi_consensus as fc
    except Exception:
        return
    orig_debug = fc.logger.debug

    def debug(msg, *a, **k):
        if isinstance(msg, str) and "[DIAG-DISTANCE]" in msg:
            m = _PAT.search(msg)
            if m:
                _ROWS.append(m.groups())
        return orig_debug(msg, *a, **k)

    fc.logger.debug = debug


def _dump():
    argv = list(sys.argv)
    if "--config" not in argv or not _ROWS:
        return
    d = os.path.dirname(os.path.abspath(argv[argv.index("--config") + 1]))
    with open(os.path.join(d, "lazy_distance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proposer", "raw_distance", "lazy_check_distance", "base_n",
                    "aggregation_refs_n", "parents_n"])
        w.writerows(_ROWS)


_install()
atexit.register(_dump)
