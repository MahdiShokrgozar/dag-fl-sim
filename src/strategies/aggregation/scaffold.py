# src/aggregation/scaffold.py (نسخه اصلاح‌شده)
import torch
import copy
from typing import Any, List, Dict, Optional, Tuple
from .base import BaseAggregator
from src.models import get_model

class ScaffoldAggregator(BaseAggregator):
    agg_type = "scaffold"

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg)
        self.device = device
        self.client_control_variates = {}

    def client_update(self, node, global_weights: Dict, server_control_variate: Dict) -> Dict:
        """
        Client update that now receives the server control variate directly from the simulator.
        """
        node_id = node.node_id
        if node_id not in self.client_control_variates:
            self.client_control_variates[node_id] = {k: torch.zeros_like(v, device=self.device) for k, v in server_control_variate.items()}
        
        c_i = self.client_control_variates[node_id]
        c_global = server_control_variate

        node.model.load_state_dict(global_weights)
        node.model.train()
        lr = node.cfg.get("local_training", {}).get("lr", 0.01)
        optimizer = torch.optim.SGD(node.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        initial_weights_for_update = copy.deepcopy(global_weights)

        for epoch in range(node.epochs):
            for data, target in node.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = node.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply SCAFFOLD gradient correction
                with torch.no_grad():
                    for param, c_global_param, c_local_param in zip(node.model.parameters(), c_global.values(), c_i.values()):
                        if param.grad is not None:
                            param.grad.add_(c_global_param.to(self.device) - c_local_param.to(self.device))
                optimizer.step()

        new_weights = node.model.state_dict()
        
        # Update local control variate (c_i) and compute its delta
        with torch.no_grad():
            c_i_new = {}
            control_delta = {}
            local_steps = len(node.train_loader) * node.epochs
            coef = 1 / (local_steps * lr)
            
            for key in c_i:
                c_i_new_val = c_i[key] - c_global[key] + (initial_weights_for_update[key] - new_weights[key]) * coef
                c_i_new[key] = c_i_new_val
                control_delta[key] = c_i_new_val - c_i[key]

            self.client_control_variates[node_id] = c_i_new

        return {"node_id": node.node_id, "weights": new_weights, "control_delta_for_server": control_delta, "model_format": "raw"}

    def aggregate(self, payloads: List[Dict], server_control_variate: Dict = None,
                  node: Any = None) -> Tuple[Dict, Dict]:
        """
        Aggregation that now takes the current server control variate and returns its updated version.

        `node` is accepted and ignored so the class satisfies the BaseAggregator
        interface, which carries it because BaseNode calls
        `aggregate(payloads, node=self)`. SCAFFOLD itself is driven by
        CentralizedSimulator, which passes the control variate positionally.
        """
        if not payloads: return {}, server_control_variate
        
        # 1. Aggregate weights
        avg_weights = {k: torch.stack([p["weights"][k].to(self.device) for p in payloads]).mean(0) for k in payloads[0]["weights"]}
        
        # 2. Update server control variate
        new_server_c = copy.deepcopy(server_control_variate)
        if all("control_delta_for_server" in p for p in payloads):
            with torch.no_grad():
                avg_control_delta = {k: torch.stack([p["control_delta_for_server"][k].to(self.device) for p in payloads]).mean(0) for k in payloads[0]["control_delta_for_server"]}
                for key in new_server_c:
                    new_server_c[key] += avg_control_delta[key]
                    
        return avg_weights, new_server_c