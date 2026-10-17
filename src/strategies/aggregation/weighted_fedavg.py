"""
Module: weighted_fedavg.py
Description:
    Weighted Federated Averaging with sample count validation.
"""

import torch
import logging
from typing import List, Dict, Optional, Any
from .fedavg import FedAvg

logger = logging.getLogger(__name__)


class WeightedFedAvg(FedAvg):
    """
    Weighted Federated Averaging.
    Performs averaging based on sample counts: sum(W_i * n_i) / sum(n_i)
    """
    agg_type = "fedavg_weighted"

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Optional[Dict]:
        """
        Weighted average based on sample counts.
        
        Args:
            payloads: List of update dictionaries with 'weights' and 'samples' keys.
            
        Returns:
            Aggregated model weights, or None if no valid payloads.
        """
        if not payloads:
            return None
        
        # 1. Filter only valid payloads with weights
        valid_payloads = [p for p in payloads if isinstance(p, dict) and "weights" in p]
        if not valid_payloads:
            return None
        
        # 2. Calculate total samples with validation
        total_samples = 0
        for p in valid_payloads:
            if "samples" not in p:
                logger.warning(
                    f"Missing 'samples' in payload from {p.get('node_id', 'unknown')}, "
                    f"using default=1"
                )
            sample_count = p.get("samples", 1)
            if sample_count <= 0:
                logger.warning(
                    f"Invalid sample count ({sample_count}) from {p.get('node_id', 'unknown')}, "
                    f"using default=1"
                )
                sample_count = 1
            total_samples += sample_count
        
        if total_samples == 0:
            logger.error("Total samples is zero, falling back to simple average")
            return FedAvg(self.cfg).aggregate(payloads)
        
        # 3. Initialize with structure of the first model
        first_weights = valid_payloads[0]["weights"]
        if "weights" in first_weights and isinstance(first_weights["weights"], dict):
            first_weights = first_weights["weights"]
        
        avg_weights = {}
        
        # 4. Iterate over parameters
        for key, val in first_weights.items():
            if isinstance(val, torch.Tensor):
                is_integral = not val.is_floating_point()
                weighted_sum = None
                
                for p in valid_payloads:
                    client_weight = p["weights"][key]
                    sample_count = p.get("samples", 1)
                    if sample_count <= 0:
                        sample_count = 1
                    
                    # Weighting factor
                    ratio = sample_count / total_samples
                    term = client_weight.float() * ratio
                    
                    if weighted_sum is None:
                        weighted_sum = term
                    else:
                        weighted_sum += term
                
                # Handle Integer Tensors (like BatchNorm counters)
                if is_integral:
                    avg_weights[key] = weighted_sum.round().long()
                else:
                    avg_weights[key] = weighted_sum
            else:
                avg_weights[key] = val
        
        return avg_weights