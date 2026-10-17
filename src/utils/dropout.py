# src/utils/dropout.py
import torch
import copy
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def apply_federated_dropout(model_state_dict: Dict[str, torch.Tensor], dropout_rate: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    Implements Generalized Federated Dropout.
    Randomly zeros out a fraction of the weights to create a sub-model, forcing
    the network to learn more robust features (Section 2.5 of DB-FL).
    
    Args:
        model_state_dict: The full model weights.
        dropout_rate: Percentage of weights to drop (0.0 to 1.0).
        
    Returns:
        A sparse representation or masked state_dict (sub-model).
    """
    if dropout_rate <= 0.0:
        return copy.deepcopy(model_state_dict)

    if dropout_rate >= 1.0:
        logger.warning(f"Dropout rate {dropout_rate} is >= 1.0. Returning heavily masked model.")

    dropped_state_dict = {}
    
    # Scale factor for Inverted Dropout to maintain expected activation magnitudes
    scale_factor = 1.0 / (1.0 - dropout_rate)
    
    for key, value in model_state_dict.items():
        # Apply dropout strictly to tensors containing weights (e.g., Conv2d, Linear)
        # We skip 1D tensors like biases or BatchNorm running stats to avoid representation collapse
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dim() > 1:
            # torch.full_like automatically matches the device (CPU/CUDA) and dtype of 'value'
            prob_tensor = torch.full_like(value, 1.0 - dropout_rate)
            mask = torch.bernoulli(prob_tensor)
            
            # Apply mask and scale (Inverted Dropout mechanism)
            dropped_state_dict[key] = value * mask * scale_factor
        else:
            # Keep biases and 1D tensors as they are
            dropped_state_dict[key] = copy.deepcopy(value)

    return dropped_state_dict