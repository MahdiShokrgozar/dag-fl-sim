# src/metrics/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class IMetric(ABC):
    """
    Abstract Base Class for all metrics.
    Ensures Strategy Pattern implementation.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the metric config."""
        pass

    @abstractmethod
    def get_global_columns(self) -> List[str]:
        """
        Returns the list of CSV columns this metric will produce.
        """
        pass
    
    @abstractmethod
    def get_node_columns(self) -> List[str]:
        """Returns the list of CSV columns for per-node details."""
        pass

    @abstractmethod
    def collect(self, node: Any) -> Dict[str, Any]:
        """
        Collects data from a single node. 
        MUST BE IDEMPOTENT (Do not reset node state here).
        """
        pass

    @abstractmethod
    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        """
        Aggregates node-level results into global metrics.
        """
        pass