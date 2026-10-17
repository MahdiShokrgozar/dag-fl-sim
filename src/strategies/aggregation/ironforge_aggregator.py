"""
IronForge Aggregation Strategy
Implements the weighted aggregation mechanism defined in Equation (1) 
of the IRONFORGE paper.
"""

import torch
import logging
from typing import List, Dict, Optional, Any
from src.strategies.aggregation.base import BaseAggregator

class IronForgeAggregator(BaseAggregator):
    """
    Aggregation strategy based on local evaluation accuracy.
    Formula: W* = \sum (E_p / \sum E_q) * W_p
    """
    agg_type = "ironforge"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device
        self.logger = logging.getLogger("IronForge.Aggregator")

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        """
        Executes local training. In IronForge, the node starts training 
        AFTER evaluating and aggregating the parent models.
        """
        if global_weights:
            try:
                node.model.load_state_dict(global_weights)
            except Exception as e:
                self.logger.error(f"Error loading global weights: {e}")
                raise e
        
        # Call the node's standard training loop
        new_weights = node.default_train_model()
        
        # We return the new weights. The metadata (like parents and timestamps)
        # will be handled by the DecentralizedSimulator engine.
        return new_weights

    def _measured_accuracy(self, outer: Dict, data: Dict, node: Any = None) -> float:
        """
        E_p for Equation (1): the evaluated accuracy of a candidate parent.

        Prefers the simulation oracle keyed by block hash -- the stand-in for the
        local evaluation pass IronForge specifies -- and falls back to the node's
        self-reported figure only when no oracle record exists (isolated tests,
        genesis). The fallback is deliberately last, not first.
        """
        claimed = float(data.get("accuracy", outer.get("accuracy", 0.0)) or 0.0)

        # Prefer the node handed in by the caller; fall back to the binding
        # BaseNode installs (`aggregator.node = self`) for callers that do
        # not pass one.
        node = node if node is not None else getattr(self, "node", None)
        sim = getattr(node, "simulator", None) if node is not None else None
        if sim is None:
            return claimed

        block_hash = outer.get("hash", outer.get("block_id", data.get("hash", "")))
        if not block_hash:
            return claimed

        record = getattr(sim, "oracle_state", {}).get(block_hash, {}) or {}
        measured = record.get("accuracy")
        return float(measured) if measured is not None else claimed

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Optional[Dict]:
        """
        Implements Equation (1) of the IRONFORGE paper.
        O(1) explicit unwrapping to handle DAG and Consensus nested payloads.
        """
        if not payloads:
            return None

        valid_payloads = []
        for p in payloads:
            if not isinstance(p, dict):
                continue
            
            # 1. Peel Layer 1: DAG Transaction Wrapper
            data = p.get("payload", p)
            
            # 2. Peel Layer 2: Consensus Wrapper (ironforge_pol)
            if isinstance(data, dict) and "payload" in data and "metadata" in data:
                data = data["payload"]
                
            # Now we are at the core BaseNode payload
            if not isinstance(data, dict):
                continue
                
            weights = data.get("weights")
            # Equation (1) weights by E_p -- the accuracy the aggregating node
            # MEASURES for each candidate parent, not the accuracy that parent
            # claims about itself. Using the claim (the previous behaviour) hands
            # the largest aggregation weight to whichever node lies hardest about
            # its own quality: the exact inversion the paper's local-evaluation
            # step exists to prevent, and one that made IronForge an artificially
            # weak baseline under poisoning.
            #
            # The measured value is read from the simulation oracle rather than by
            # running a real evaluation pass, consistent with how PoL is handled in
            # ironforge_vrf.py: the simulator models the cost instead of paying it.
            # The matching inference cost is already charged in
            # IronForgeTipSelector (`evaluation_time_per_model` x sample_size), so
            # changing the VALUE here leaves cost accounting untouched -- it only
            # stops the value from being self-reported.
            # See thesis-hub/docs/decisions.md, 2026-09-12.
            acc = self._measured_accuracy(p, data, node)

            if weights is not None and float(acc) > 0.0:
                valid_payloads.append({
                    "weights": weights,
                    "accuracy": float(acc)
                })
        
        if not valid_payloads:
            # Using debug instead of warning to prevent log spam if early epochs legitimately lack accuracy
            self.logger.debug("No valid payloads with accuracy scores found for IronForge aggregation.")
            return None

        # Calculate the denominator: \sum E_q
        total_accuracy = sum(p["accuracy"] for p in valid_payloads)

        # Edge case: if all accuracies are 0.0 (e.g., initial untrained models)
        if total_accuracy <= 0.0:
            self.logger.debug("Total accuracy is zero. Falling back to simple average.")
            weights_factor = [1.0 / len(valid_payloads) for _ in valid_payloads]
        else:
            # Calculate the weighting factor: E_p / \sum E_q
            weights_factor = [p["accuracy"] / total_accuracy for p in valid_payloads]

        # Perform the weighted aggregation (Tensor operations)
        first_weights = valid_payloads[0]["weights"]
        aggregated_weights = {}

        for key, val in first_weights.items():
            if isinstance(val, torch.Tensor):
                is_integral = not val.is_floating_point()
                weighted_sum = None

                for idx, payload in enumerate(valid_payloads):
                    client_weight_tensor = payload["weights"][key]
                    factor = weights_factor[idx]
                    
                    term = client_weight_tensor * factor
                    if weighted_sum is None:
                        weighted_sum = term
                    else:
                        weighted_sum += term

                if is_integral:
                    aggregated_weights[key] = weighted_sum.round().long()
                else:
                    aggregated_weights[key] = weighted_sum
            else:
                aggregated_weights[key] = val

        return aggregated_weights