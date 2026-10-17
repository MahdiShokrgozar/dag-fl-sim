# src/metrics/collectors/backdoor_asr.py
from types import SimpleNamespace
from typing import List, Dict, Any

import numpy as np

from ..interface import IMetric
from src.nodes.malicious.backdoor import apply_trigger, backdoor_settings
from src.utils.fast_loader import FastTensorDataLoader


class BackdoorASRMetric(IMetric):
    """
    Backdoor attack success rate (2026-09-26, thesis-hub/docs/decisions.md same
    date): the share of held-out test images whose true label is NOT the
    target that a node's current model classifies as the target once the
    attacker's trigger is stamped on them (Bagdasaryan et al. 2020, "backdoor
    accuracy"). Main-task accuracy is AccuracyMetric's held-out column; a
    backdoor is only meaningful read against both.

    Protocol-agnostic: it evaluates every node's weights the same way for every
    protocol. It only runs when the population contains backdoor attackers or
    participants.attacks.backdoor.measure_asr is set (clean-model floor);
    otherwise the columns are empty and nothing is evaluated.
    """

    @property
    def name(self) -> str:
        return "backdoor_asr"

    def get_global_columns(self) -> List[str]:
        return ["backdoor_asr_honest_mean", "backdoor_asr_honest_max"]

    def get_node_columns(self) -> List[str]:
        return ["node_backdoor_asr"]

    @staticmethod
    def _active(node: Any) -> bool:
        participants = node.config.get("participants", {})
        dist = participants.get("distribution", {}) or {}
        bd_cfg = (participants.get("attacks", {}) or {}).get("backdoor", {}) or {}
        # measure_asr: evaluate even with no attacker, for the clean-model floor.
        return float(dist.get("backdoor", 0.0) or 0.0) > 0.0 or bool(bd_cfg.get("measure_asr", False))

    @staticmethod
    def _asr_loader(node: Any):
        sim = getattr(node, "simulator", None)
        cached = getattr(sim, "_backdoor_asr_loader", None)
        if cached is not None:
            return cached
        base = getattr(sim, "global_test_loader", None)
        if base is None:
            return None
        trigger_size, target_label, _ = backdoor_settings(node.config)
        keep = base.dataset_targets != target_label
        data = apply_trigger(base.dataset_data[keep].clone().float(), trigger_size)
        targets = base.dataset_targets[keep].clone()
        targets[:] = target_label
        loader = FastTensorDataLoader(
            SimpleNamespace(data=data, targets=targets),
            batch_size=base.batch_size, shuffle=False, device=base.device, cache_on_device=False,
        )
        sim._backdoor_asr_loader = loader
        return loader

    def collect(self, node: Any) -> Dict[str, Any]:
        row = {"node_backdoor_asr": float("nan"),
               "_asr_is_honest": not bool(getattr(node, "is_malicious", False))}
        if not self._active(node) or not hasattr(node, "cache_manager") or not hasattr(node, "trainer"):
            return row
        loader = self._asr_loader(node)
        if loader is None:
            return row
        try:
            model = node.cache_manager.checkout_model(node.weights)
            asr, _ = node.trainer.evaluate_state(model, loader=loader)
            node.cache_manager.checkin_model(model)
            row["node_backdoor_asr"] = float(asr)
        except Exception:
            pass
        return row

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        vals = [r["node_backdoor_asr"] for r in results
                if r.get("_asr_is_honest") and r.get("node_backdoor_asr") == r.get("node_backdoor_asr")]
        if not vals:
            return {"backdoor_asr_honest_mean": float("nan"), "backdoor_asr_honest_max": float("nan")}
        return {"backdoor_asr_honest_mean": float(np.mean(vals)),
                "backdoor_asr_honest_max": float(np.max(vals))}
