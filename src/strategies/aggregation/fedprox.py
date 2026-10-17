# src/aggregation/fedprox.py
import torch
import copy
from typing import List, Dict, Optional, Any
from .base import BaseAggregator

class FedProx(BaseAggregator):
    """
    Federated Proximal (FedProx) Optimization.
    Adds a proximal term to the local loss to limit divergence from global model.
    """
    agg_type = "fedprox"

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")):
        super().__init__(cfg)
        self.device = device
        self.mu = cfg.get("algorithm", {}).get("fedprox_mu", 0.01)

    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        """
        Custom training loop including the proximal term.
        """
        # 1. Load Global Model
        if global_weights:
            node.model.load_state_dict(global_weights)
        
        # Keep a copy of the global model for the proximal term calculation
        global_model = copy.deepcopy(node.model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        # 2. Setup Optimizer
        node.model.train()
        optimizer = torch.optim.SGD(
            node.model.parameters(), 
            lr=self.cfg.get("local_training", {}).get("lr", 0.01)
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 3. Proximal Training Loop
        epochs = self.cfg.get("local_training", {}).get("epochs", 1)
        
        for _ in range(epochs):
            for data, target in node.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                output = node.model(data)
                loss = criterion(output, target)
                
                # Add Proximal Term: (mu / 2) * ||w - w_global||^2
                prox_term = 0.0
                for w, w_t in zip(node.model.parameters(), global_model.parameters()):
                    prox_term += (w - w_t).norm(2)
                
                loss += (self.mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()

        return {
            "node_id": node.node_id,
            "weights": node.model.state_dict(),
            "model_format": "raw"
        }

    def aggregate(self, payloads: List[Dict], node: Any = None) -> Dict:
        """
        Standard averaging (FedProx modifies training, not aggregation).
        """
        if not payloads: return {}
        valid = [p for p in payloads if "weights" in p]
        if not valid: return {}
        
        avg_weights = {}
        first_keys = valid[0]["weights"].keys()
        n = len(valid)
        
        for key in first_keys:
            stacked = torch.stack([p["weights"][key] for p in valid])
            avg_weights[key] = stacked.sum(0) / n
            
        return avg_weights