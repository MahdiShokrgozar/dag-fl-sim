"""
Module: trimmed_mean.py
Description:
    Coordinate-wise Trimmed Mean Aggregation with device safety.
    Removes the largest and smallest values (outliers) before averaging.
"""

import torch
import logging
from typing import List, Dict, Optional, Any
from .base import BaseAggregator
from .fedavg import FedAvg  # Moved to top-level import

logger = logging.getLogger(__name__)


class TrimmedMean(BaseAggregator):
    """
    Coordinate-wise Trimmed Mean Aggregation.
    Removes the largest and smallest values (outliers) before averaging.
    """
    agg_type = "trimmed_mean"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device
        # Beta is the fraction to trim from each end (e.g., 0.1 for 10%)
        self.beta = cfg.get("algorithm", {}).get("trimmed_beta", 0.1)

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        if global_weights:
            node.model.load_state_dict(global_weights)
        new_weights = node.default_train_model()
        return {"node_id": node.node_id, "weights": new_weights}

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Dict:
        """
        Trimmed Mean aggregation with device safety.
        
        Args:
            payloads: List of update dictionaries with 'weights' key.
            
        Returns:
            Aggregated model weights.
        """
        if not payloads:
            return {}
        
        valid = [p for p in payloads if "weights" in p]
        n = len(valid)
        if not valid:
            return {}
        
        # Calculate how many to trim
        trim_cnt = int(n * self.beta)
        if n - 2 * trim_cnt <= 0:
            # Fallback to simple average if trim is too aggressive
            logger.warning(
                f"Trim count too aggressive (n={n}, trim_cnt={trim_cnt}), "
                f"falling back to FedAvg"
            )
            return FedAvg(self.cfg).aggregate(payloads)
        
        agg_weights = {}
        first_keys = valid[0]["weights"].keys()
        
        for key in first_keys:
            # CRITICAL FIX: Ensure all tensors on same device
            device = valid[0]["weights"][key].device
            stacked = torch.stack([p["weights"][key].to(device) for p in valid])
            
            # Sort along client dimension
            sorted_tensor, _ = torch.sort(stacked, dim=0)
            
            # Slice to remove top and bottom outliers
            trimmed = sorted_tensor[trim_cnt : n - trim_cnt]
            
            # Average the rest
            agg_weights[key] = trimmed.mean(dim=0)
        
        return agg_weights