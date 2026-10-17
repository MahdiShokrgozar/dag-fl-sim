"""
Module: fedavg.py
Description:
    Simple Federated Averaging with device safety and integer tensor handling.
"""

import torch
import logging
from typing import List, Dict, Optional, Any
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvg(BaseAggregator):
    """
    Simple Federated Averaging.
    Performs simple unweighted averaging: sum(W_i) / N
    """
    agg_type = "fedavg_simple"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        """
        Standard local training via the node's LocalTrainer.

        Only reachable from CentralizedSimulator today (no decentralized path
        calls client_update). Rewritten 2026-09-13 (thesis-hub/docs/
        decisions.md, same date): this used to call node.model.load_state_dict
        and node.default_train_model(), neither of which exist on BaseNode -
        that API predates the cache_manager/LocalTrainer refactor and this
        path was never exercised to catch the drift. train_and_evaluate()
        already checks out a model from the node's cache_manager, trains it,
        evaluates it, and returns plain CPU tensors - exactly what aggregate()
        below expects in a payload's "weights" dict.
        """
        if global_weights and "weights" in global_weights and isinstance(global_weights["weights"], dict):
            global_weights = global_weights["weights"]

        init_weights = global_weights if global_weights else node.weights
        new_weights, compute_time, _accuracy = node.trainer.train_and_evaluate(init_weights)
        node.telemetry.register_compute(compute_time)
        node.weights = new_weights

        return {"weights": new_weights, "model_format": "raw"}

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Optional[Dict]:
        """
        Computes Simple Average.
        Formula: W_global = sum(W_k) / K
        
        Handles:
        - Device mismatches (CPU/GPU)
        - Integer tensors (BatchNorm running stats)
        - Shape inconsistencies
        
        Args:
            payloads: List of update dictionaries with 'weights' key.
            
        Returns:
            Aggregated model weights, or None if no valid payloads.
        """
        if not payloads:
            return None
        
        # 1. Filter only valid payloads with weights
        valid_payloads = [p for p in payloads if isinstance(p, dict) and "weights" in p]
        if not valid_payloads:
            return None
        
        num_models = len(valid_payloads)
        
        # 2. Initialize with structure of the first model
        first_weights = valid_payloads[0]["weights"]
        if "weights" in first_weights and isinstance(first_weights["weights"], dict):
            first_weights = first_weights["weights"]
        
        avg_weights = {}
        
        # 3. Iterate over parameters
        for key, val in first_weights.items():
            # Only process Tensors
            if isinstance(val, torch.Tensor):
                # Check if integer tensor (e.g., BatchNorm running_mean)
                is_integral = not val.is_floating_point()
                
                # Start with a clone of the first model's weight (convert to float)
                sum_tensor = val.clone().float()
                
                # Add others
                for i in range(1, num_models):
                    other_tensor = valid_payloads[i]["weights"][key]
                    
                    # Ensure same device
                    if other_tensor.device != sum_tensor.device:
                        other_tensor = other_tensor.to(sum_tensor.device)
                    
                    # Check shape consistency
                    if other_tensor.shape != sum_tensor.shape:
                        logger.error(
                            f"Shape mismatch for {key}: {sum_tensor.shape} vs {other_tensor.shape}. "
                            f"Skipping this parameter."
                        )
                        break
                    
                    sum_tensor += other_tensor.float()
                else:
                    # Only execute if no break occurred
                    # Divide by number of models
                    avg_tensor = sum_tensor / num_models
                    
                    # Convert back to original dtype if integer
                    if is_integral:
                        avg_tensor = avg_tensor.round().to(val.dtype)
                    
                    avg_weights[key] = avg_tensor
            else:
                # For non-tensors (scalars), keep first
                avg_weights[key] = val
        
        return avg_weights