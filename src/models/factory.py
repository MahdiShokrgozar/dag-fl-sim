# src/models/factory.py

from typing import Dict, Any
import torch.nn as nn
from .cnn import MNISTCNN, CIFARCNN, CIFARCNNGN

class ModelFactory:
    """
    Factory class to instantiate neural network models based on configuration.
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        """
        Creates and returns a model instance.
        
        Args:
            config: Global configuration dictionary.
            
        Returns:
            nn.Module: The instantiated PyTorch model.
            
        Raises:
            ValueError: If the model name in config is unknown.
        """
        model_cfg = config.get("model", {})
        name = model_cfg.get("architecture", model_cfg.get("name", "cnn_mnist")).lower()

        if name in ["cnn_mnist", "cnn"]:
            return MNISTCNN()
        elif name == "cnn_cifar10":
            return CIFARCNN()
        elif name == "cnn_cifar10_gn":
            return CIFARCNNGN()
        elif name == "mlp":
            # return MNISTMLP()
            raise NotImplementedError("MLP model not yet implemented.")
        else:
            raise ValueError(f"Unknown model name specified in config: '{name}'")