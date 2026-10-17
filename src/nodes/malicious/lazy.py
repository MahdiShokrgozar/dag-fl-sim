import copy
import logging
from typing import Dict, Any, Tuple

import torch

from src.nodes.base_node import BaseNode
from src.nodes.malicious.base_malicious import MaliciousTrainer

logger = logging.getLogger("Node.Lazy")


def add_gaussian_noise(weights: Dict[str, Any], noise_std: float) -> Dict[str, Any]:
    """Copy of `weights` with N(0, noise_std^2) added to every floating-point tensor."""
    out = {}
    for k, v in weights.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v + (torch.randn_like(v) * noise_std).to(v.device)
        else:
            out[k] = copy.deepcopy(v)
    return out

class LazyTrainer(MaliciousTrainer):
    """
    Bypasses the local training phase entirely and returns the aggregated parent models 
    as its own work to conserve resources.
    """
    
    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        self.node.logger.info(f"[{self.node.node_id}] Executing LAZY attack: bypassing training.")
        
        lazy_weights = copy.deepcopy(init_weights)
        fake_compute_time = 0.01
        
        # True accuracy is exactly the accuracy of the base model
        true_accuracy = getattr(self.node, 'last_accuracy', 0.0)

        # Noisy lazy (thesis-hub/docs/decisions.md, 2026-09-30, in-band attackers):
        # republish the base plus Gaussian noise so the exact-zero lazy gate no
        # longer fires. noise_std is per parameter, as in the poisoning attacker.
        # Default 0.0 draws no random numbers, so the original attack is unchanged.
        atk_cfg = self.node.config.get("participants", {}).get("attacks", {}).get("lazy", {}) or {}
        noise_std = float(atk_cfg.get("noise_std", 0.0))
        if noise_std > 0.0:
            lazy_weights = add_gaussian_noise(lazy_weights, noise_std)
            if bool(atk_cfg.get("evaluate_true_accuracy", False)):
                model = self.node.cache_manager.checkout_model(lazy_weights)
                true_accuracy, _ = self.evaluate_state(model)
                self.node.cache_manager.checkin_model(model)
        
        # Spoofing to claim improvements
        self.spoof_metadata(target_accuracy=93.5)
        
        return lazy_weights, fake_compute_time, true_accuracy

class LazyNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = LazyTrainer(self)