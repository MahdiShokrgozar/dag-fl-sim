# tests/conftest.py

import pytest
import warnings
import torch
import sys
import os
from unittest.mock import MagicMock

# Ensure the 'src' module is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def suppress_annoying_warnings():
    """
    Suppresses specific warnings from third-party libraries like Torchvision
    to keep test output clean and fast.
    """
    # فیلتر کردن وارنینگ خاص Pillow که مربوط به پارامتر mode است
    warnings.filterwarnings("ignore", message=".*mode.*parameter is deprecated.*")
    
    # یا فیلتر کردن کلی وارنینگ‌های Deprecation مربوط به torchvision
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
    
@pytest.fixture
def mock_config():
    """
    Provides a standardized configuration dictionary for testing.
    
    Returns:
        dict: A default configuration structure mimicking 'defaults.yaml'.
    """
    return {
        "loglevel": "DEBUG",
        "simulation": {
            "type": "decentralized",
            "protocol": "generic_configurable",
            "seed": 42,
            "device": "cpu",
            "max_time": 10.0,
        },
        "strategy_config": {
            "aggregator": "fedavg",
            "tip_selector": "random",
            "consensus": "simple",
            "l2_strategy": "none",
            "batching": {"mode": "size", "threshold": 2},
            "processing": "identity"
        },
        "strategy": {
            "cluster_size": 5,
            "krum_f": 1
        },
        "network": {
            "num_nodes": 3,
            "latency": {"min": 0.01, "max": 0.05, "jitter": 0.0}
        },
        "participants": {
            "total": 3,
            "malicious_rate": 0.0,
            "adversary_type": "poisoning"
        },
        "model": {
            "name": "cnn",
            "num_classes": 10
        },
        "dataset": {
            "name": "mnist",
            "iid": True
        },
        "straggler": {
            "enabled": False
        },
        "reporting": {
            "interval": 1.0,
            "save_dir": "./tests/output",
            "metrics": {"accuracy": False}
        },
        "storage": {
            "type": "mock"
        }
    }

@pytest.fixture
def simple_model():
    """
    Creates a lightweight PyTorch model for unit testing.
    
    Returns:
        torch.nn.Module: A simple Linear model.
    """
    return torch.nn.Linear(10, 2)

@pytest.fixture
def mock_simulator(mock_config):
    """
    Creates a Mock Simulator object with basic attributes.
    """
    sim = MagicMock()
    sim.current_time = 0.0
    sim.broadcast = MagicMock()
    return sim