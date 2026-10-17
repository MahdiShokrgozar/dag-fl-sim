# src/utils/gpu_cache.py

import torch
import copy
import logging
from typing import Dict, Any, Optional

class ModelCacheManager:
    """
    Manages a pool of shared model instances to optimize RAM and VRAM usage.
    Implements the 'Object Pool' pattern.
    
    Instead of every node holding its own model instance (which consumes huge RAM),
    nodes hold only lightweight weights (Dict). When training is needed, 
    they 'checkout' a model from this pool, load their weights, train, 
    and then 'checkin' the model back to the pool.
    """
    def __init__(self, template_model: torch.nn.Module, device: str = "cuda", max_capacity: int = 5):
        self.device = torch.device(device)
        self.max_capacity = max_capacity
        self.logger = logging.getLogger("ModelCache")
        
        # Keep the template on CPU to create new instances if needed
        self.template = copy.deepcopy(template_model).to("cpu")
        
        # The pool of active models residing on the target device (e.g., GPU)
        self.pool = []
        
        # Pre-fill the pool
        try:
            for _ in range(max_capacity):
                m = copy.deepcopy(self.template).to(self.device)
                m.eval() # Default state
                self.pool.append(m)
            self.logger.info(f"✅ Model Cache initialized with {max_capacity} instances on {self.device}.")
        except RuntimeError as e:
            self.logger.warning(f"⚠️ Could not pre-fill GPU cache (OOM?): {e}. Will create on demand.")

    def checkout_model(self, state_dict: Dict[str, Any]) -> torch.nn.Module:
        """
        Borrows a model from the pool and loads the provided weights into it.
        """
        if not self.pool:
            # If pool is empty (all models are busy), create a temporary one.
            # This handles edge cases where parallelism exceeds pool size.
            model = copy.deepcopy(self.template).to(self.device)
        else:
            model = self.pool.pop()
            
        # Load the specific node's weights onto the shared model
        # Note: state_dict is expected to be on CPU, load_state_dict handles the move.
        model.load_state_dict(state_dict)
        return model

    def checkin_model(self, model: torch.nn.Module):
        """
        Returns a model to the pool after use.
        """
        # Clear gradients to save memory and prevent leakage between nodes
        for param in model.parameters():
            param.grad = None
            
        if len(self.pool) < self.max_capacity:
            self.pool.append(model)
        else:
            # If pool is full (e.g., returned a temp model), discard it
            del model
            
    def get_template(self):
        return self.template