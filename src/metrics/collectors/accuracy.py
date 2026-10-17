# src/metrics/collectors/accuracy.py
import json
import numpy as np
from typing import List, Dict, Any
from ..interface import IMetric

class AccuracyMetric(IMetric):
    """
    Collects and aggregates accuracy metrics across the federated network.
    
    In DAG-based Asynchronous FL, simple averaging causes artificial performance 
    degradation due to stale tips from straggler nodes. This collector implements 
    a 'Reputation-Weighted Aggregation' strategy. It prioritizes the State-of-the-Art 
    knowledge present at the active frontier of the DAG.
    """

    @property
    def name(self) -> str: 
        return "accuracy"

    def get_global_columns(self) -> List[str]:
        return ["global_mean_accuracy", "global_max_accuracy", "global_held_out_accuracy"]

    def get_node_columns(self) -> List[str]:
        return [
            "node_accuracy",
            "per_class_acc_json",
            "reputation_weight",
            "is_active",
            "node_global_held_out_accuracy"
        ]

    def collect(self, node: Any) -> Dict[str, Any]:
        """
        Extracts local accuracy and generic reputation metadata from a node.
        
        Args:
            node: The node instance (must inherit from BaseNode).
            
        Returns:
            Dict containing the collected metrics and weights for this node.
        """
        stats = node.snapshot_metrics()
        
        # 1. Extract model accuracy
        acc = stats.get("node_accuracy", 0.0)
        if acc == 0.0:
            acc = getattr(node, "last_accuracy", 0.0)

        if hasattr(acc, 'item'):
            acc = acc.item()

        # 2. Extract reputation agnostically (Protocol-independent)
        # Defaults to 1.0 (equivalent to standard FedAvg) if the protocol 
        # lacks a reputation system.
        rep_weight = 1.0
        if hasattr(node, 'get_current_reputation'):
            try:
                rep_weight = float(node.get_current_reputation())
            except (ValueError, TypeError):
                rep_weight = 1.0

        # 3. Serialize per-class accuracy
        per_class_raw = stats.get("per_class_accuracy", getattr(node, "per_class_accuracy", {}))
        per_class_json = json.dumps({str(k): float(v) for k, v in per_class_raw.items()}) if per_class_raw else "{}"

        # 4. Global held-out accuracy - REPORTING ONLY, never fed into any
        # protocol's tip-selection/PoL/reputation decision. node_accuracy
        # above (since 2026-09-15) reflects this node's own narrow local
        # test partition, which "looks easier" under severe non-IID purely
        # because it's a smaller label set - reporting only that number
        # would hide whether the model generalizes at all, the same failure
        # mode in the opposite direction from the global-test-set bug this
        # local partition itself fixed. Evaluates THIS node's current
        # weights against the untouched global_test_set that
        # DecentralizedSimulator keeps for exactly this purpose (absent on
        # CentralizedSimulator, which has no need for it - a real central
        # server's one model IS the global-view number already).
        # See thesis-hub/docs/decisions.md, 2026-09-15.
        global_held_out_acc = None
        global_loader = getattr(getattr(node, "simulator", None), "global_test_loader", None)
        if global_loader is not None and hasattr(node, "cache_manager") and hasattr(node, "trainer"):
            try:
                eval_model = node.cache_manager.checkout_model(node.weights)
                global_held_out_acc, _ = node.trainer.evaluate_state(eval_model, loader=global_loader)
                node.cache_manager.checkin_model(eval_model)
            except Exception:
                global_held_out_acc = None

        return {
            "node_accuracy": float(acc),
            "per_class_acc_json": per_class_json,
            "reputation_weight": rep_weight,
            "is_active": 0 if stats.get("is_slashed") else 1,
            "node_global_held_out_accuracy": float(global_held_out_acc) if global_held_out_acc is not None else float(acc)
        }

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        """
        Calculates the global accuracy using Reputation-Weighted Averaging.
        
        Args:
            results: List of dictionaries collected from active nodes.
            global_state: Optional simulator state context.
            
        Returns:
            Dict containing the aggregated global accuracy metrics.
        """
        # Exclude slashed or inactive nodes from the aggregation
        valid_results = [r for r in results if r.get("is_active", 0) == 1]

        if not valid_results:
            return {"global_mean_accuracy": 0.0, "global_max_accuracy": 0.0, "global_held_out_accuracy": 0.0}

        # Extract arrays for vectorized computation
        accs = np.array([r["node_accuracy"] for r in valid_results])
        weights = np.array([r.get("reputation_weight", 1.0) for r in valid_results])

        # Safety Check 1: Prevent negative weights (e.g., from severe slashing penalties)
        weights = np.maximum(weights, 0)

        # Safety Check 2: Prevent division by zero if all weights fall to zero
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)

        # Compute the DAG-aware weighted accuracy
        weighted_mean_acc = np.average(accs, weights=weights)

        # Same reputation-weighted average, but over each node's accuracy
        # against the untouched global test set instead of its own local
        # partition - see the matching comment in collect(). Report both:
        # a large gap between the two IS the finding under severe non-IID,
        # not noise to be averaged away.
        held_out_accs = np.array([r.get("node_global_held_out_accuracy", r["node_accuracy"]) for r in valid_results])
        weighted_held_out_acc = np.average(held_out_accs, weights=weights)

        return {
            "global_mean_accuracy": float(weighted_mean_acc),
            "global_max_accuracy": float(np.max(accs)),
            "global_held_out_accuracy": float(weighted_held_out_acc)
        }