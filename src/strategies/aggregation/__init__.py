# src/aggregation/__init__.py

from .base import BaseAggregator
from .fedavg import FedAvg
from .fedprox import FedProx
from .krum import Krum
from .median import Median
from .trimmed_mean import TrimmedMean

# Note: Scaffold needs stateful clients, omitted for simple DAG compatibility
# from .scaffold import Scaffold 

__all__ = [
    "BaseAggregator",
    "FedAvg",
    "FedProx",
    "Krum",
    "Median",
    "TrimmedMean"
]