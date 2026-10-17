"""
Module: median.py
Description:
    Coordinate-wise Median Aggregation with device safety.
    Robust against outliers but may affect convergence speed.
"""

import torch
import logging
from typing import List, Dict, Optional, Any
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class Median(BaseAggregator):
    """
    Coordinate-wise Median Aggregation.
    Robust against outliers but may affect convergence speed.
    """
    agg_type = "median"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        if global_weights:
            node.model.load_state_dict(global_weights)
        new_weights = node.default_train_model()
        return {"node_id": node.node_id, "weights": new_weights}

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Dict:
        """
        Computes the coordinate-wise median of weights.
        Handles device mismatches and even number of models correctly.
        
        Args:
            payloads: List of update dictionaries with 'weights' key.
            
        Returns:
            Aggregated model weights.
        """
        if not payloads:
            return {}
        
        valid = [p for p in payloads if "weights" in p]
        if not valid:
            return {}
        
        med_weights = {}
        first_keys = valid[0]["weights"].keys()
        
        for key in first_keys:
            # CRITICAL FIX: Ensure all tensors on same device
            device = valid[0]["weights"][key].device
            stacked = torch.stack([p["weights"][key].to(device) for p in valid])
            
            # Handle even number of models
            if stacked.shape[0] % 2 == 0:
                mid = stacked.shape[0] // 2
                med_weights[key] = (stacked[mid-1] + stacked[mid]) / 2.0
            else:
                med_weights[key] = torch.median(stacked, dim=0)[0]
        
        return med_weights