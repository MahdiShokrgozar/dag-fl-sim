# src/aggregation/krum.py
import torch
from typing import List, Dict, Optional, Any
from .base import BaseAggregator

class Krum(BaseAggregator):
    """
    Krum Aggregation Rule.
    Selects one local model that minimizes the sum of Euclidean distances 
    to its nearest neighbors. Robust against Byzantine attacks.
    """
    agg_type = "krum"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        # Standard local training
        if global_weights:
            node.model.load_state_dict(global_weights)
        new_weights = node.default_train_model()
        return {"node_id": node.node_id, "weights": new_weights}

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Dict:
        """
        Selects the single 'best' model based on Krum distance metric.
        """
        if not payloads: return {}
        valid = [p for p in payloads if "weights" in p]
        n = len(valid)
        if n < 3: 
            # Fallback to average if not enough nodes for Krum logic
            from .fedavg import FedAvg
            return FedAvg(self.cfg).aggregate(payloads)

        # Assuming f < n/2 - 2 Byzantine nodes. k = n - f - 2
        # Simplified: We look at closest n-2 neighbors
        k = max(1, n - 2) 
        
        scores = []
        
        # Flatten weights for distance calculation
        # This can be heavy, but necessary for Krum
        flattened = []
        for p in valid:
            flat = torch.cat([p["weights"][k].flatten() for k in p["weights"]])
            flattened.append(flat)
            
        # Calculate distances
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j: continue
                d = torch.norm(flattened[i] - flattened[j]).item()
                dists.append(d)
            
            # Sum of distances to k nearest neighbors
            dists.sort()
            score = sum(dists[:k])
            scores.append(score)
            
        # Select the index with minimal score
        best_idx = scores.index(min(scores))
        return valid[best_idx]["weights"]