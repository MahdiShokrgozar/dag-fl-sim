from abc import ABC, abstractmethod
from typing import Dict, Any

class ConsensusStrategy(ABC):
    """
    Abstract Base Class for consensus mechanisms (e.g., PoW, PoL, Voting).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """
        Wraps the payload with consensus metadata (e.g., nonce, signature).
        """
        return {"metadata": "default", "proposer": node_id}

    @abstractmethod
    def validate_block(self, block: Dict[str, Any]) -> bool:
        """
        Validates a received block based on consensus rules.
        """
        return True