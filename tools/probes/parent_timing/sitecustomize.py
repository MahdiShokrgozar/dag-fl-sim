"""
Parent-timing probe (plan 44, thesis-hub/docs/decisions.md 2026-09-25).

Question: before the L2 timing fix (3509f5b), did FARABI nodes aggregate
parents whose model did not exist yet in simulated time? An L2 block's
`timestamp` is its block_ready_time (base_node.py), so a parent selected at
time `now` with timestamp > now was used before it was ready - information
from the future. If that happened often, the old-code accuracy (91.5% on
P13, 64.5% on CIFAR GN) is an inflated reference, not a target.

How: this file is put on PYTHONPATH together with the repo root. Python
imports sitecustomize at interpreter start, including the main.py
subprocesses run_experiments.py spawns, so every run is instrumented without
touching the simulator. It wraps FarabiTipSelector.select_tips_for_aggregation
and, for each selected parent, records its timestamp against the current
simulated time. Output: one CSV per run, next to the run's metrics
(<run_dir>/parent_timing.csv), written at process exit.

Works on both testOnServer (old timing) and fix/l2-block-ready-timing.
"""

import atexit
import csv
import os
import sys

_ROWS = []


def _run_dir():
    argv = list(sys.argv)
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            return os.path.dirname(os.path.abspath(argv[i + 1]))
    return None


def _lookup(node, h):
    for store in (
        getattr(getattr(node, "l2_strategy", None), "blocks", None),
        getattr(getattr(node, "local_dag", None), "vertices", None),
        getattr(getattr(getattr(node, "simulator", None), "dag", None), "vertices", None),
    ):
        if isinstance(store, dict) and h in store:
            return store[h]
    return None


def _install():
    try:
        from src.strategies.selection import farabi_selection as fs
    except Exception:
        return
    orig = fs.FarabiTipSelector.select_tips_for_aggregation
    if getattr(orig, "_probe_wrapped", False):
        return

    def wrapped(self, dag, node, current_time):
        result = orig(self, dag, node, current_time)
        try:
            ids = result[0] if isinstance(result, tuple) else result
            now = float(getattr(node.simulator, "current_time", 0.0))
            local_view_tips = 0
            l2 = getattr(node, "l2_strategy", None)
            if l2 is not None and hasattr(l2, "get_local_view"):
                try:
                    local_view_tips = len(l2.get_local_view(node.node_id, now)[0])
                except Exception:
                    local_view_tips = -1
            for h in ids:
                blk = _lookup(node, h)
                if not isinstance(blk, dict):
                    continue
                ts = blk.get("timestamp")
                if ts is None:
                    continue
                author = blk.get("author", blk.get("proposer", ""))
                _ROWS.append((
                    round(now, 4), node.node_id, h, author, round(float(ts), 4),
                    round(float(ts) - now, 4), len(ids), local_view_tips,
                    int(bool(blk.get("hash", h).startswith("00000000"))),
                ))
        except Exception:
            pass
        return result

    wrapped._probe_wrapped = True
    fs.FarabiTipSelector.select_tips_for_aggregation = wrapped


def _dump():
    d = _run_dir()
    if not d or not _ROWS:
        return
    with open(os.path.join(d, "parent_timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["now", "node", "parent", "parent_author", "parent_ts",
                    "lead", "n_parents", "local_view_tips", "is_genesis"])
        w.writerows(_ROWS)


# sys.argv may not be populated yet while site runs, so install always (only
# FarabiTipSelector is touched) and resolve the run directory at exit.
_install()
atexit.register(_dump)
