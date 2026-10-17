"""
Module: signal_probe.py
Description:
    Lightweight, event-level signal recorder for FARABI consensus analysis.
"""

import csv
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class SignalProbe:
    """
    A minimal CSV-backed recorder for per-validation consensus signals.
    """

    # Kept in sync with TrustSignalExtractor.extract(). c6_self_jump,
    # layer_drift_var and sparsity_ratio were dropped on 2026-09-05 (redundant
    # with c1 at rho~0.9, and each an extra O(P) pass) — leaving them listed here
    # would emit permanently empty columns. Group-1 CSVs written before that date
    # still carry them, and the analysis scripts tolerate missing columns.
    FIELDS: List[str] = [
        "t", "validator", "proposer", "label",
        "c1_rel_l2", "c2_cos", "c4_sign", "c5_fp",
        "c7_delta_acc", "c8_m3",
        "class_entropy", "staleness",
        "us_c1", "us_c2",
    ]

    def __init__(self, node_id: str, enabled: bool, output_dir: str = ".") -> None:
        self.node_id = node_id
        self.enabled = enabled
        self._file = None
        self._writer = None
        
        if enabled:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"probe_{node_id}.csv")
            self._file = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.FIELDS)
            self._file.flush()
            logger.info(f"[SIGNAL_PROBE] Enabled for {node_id} -> {path}")

    def record(self, row: Dict[str, Any]) -> None:
        """Appends one observation row and flushes immediately."""
        if not self.enabled or self._writer is None:
            return
        self._writer.writerow([row.get(field, "") for field in self.FIELDS])
        self._file.flush()

    def close(self) -> None:
        """Flushes remaining buffer and releases the underlying file handle."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
            
    def __del__(self):
        self.close()