# src/strategies/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAggregator(ABC):
    """
    Abstract Base Class for model aggregation strategies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates a list of model updates into a single global state.
        
        Args:
            updates: A list of dictionaries containing 'weights' and metadata.
            
        Returns:
            A dictionary representing the aggregated model weights.
        """
        pass

class L2Strategy(ABC):
    """
    Abstract Base Class for Layer-2 topology management.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def manage_topology(self, nodes: Dict[str, Any], current_time: float) -> None:
        """
        Dynamically reorganizes the network topology (e.g., forming clusters).
        Should modify 'node.l2_state' directly.
        """
        pass

class TipSelector(ABC):
    """
    Abstract Base Class for DAG tip selection.
    """
    @abstractmethod
    def select_tips(self, dag: Any, node_id: str, k: int = 2) -> List[str]:
        """
        Selects parent transactions for a new block.
        """
        pass
    
class Reputation(ABC):
    """
    Abstract Base Class for Reputation Systems.
    """
    def update(self, node_id: str, metric_data: Dict[str, Any]):
        pass

    def get_score(self, node_id: str) -> float:
        return 1.0

class BatchPolicy(ABC):
    """
    Abstract Base Class for Batching Policies.
    """
    def should_batch(self, pending_updates: List[Any], current_time: float = 0.0) -> bool:
        return True

class UpdateProcessor(ABC):
    """
    Abstract Base Class for Update Processing (Compression/Privacy).
    """
    def process_outbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    def process_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

class ConsensusAdapter(ABC):
    """
    Abstract Base Class for Consensus Logic.
    """
    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        return {"payload": payload, "metadata": {"type": "default"}}

    def validate_block(self, block: Dict[str, Any]) -> bool:
        return True