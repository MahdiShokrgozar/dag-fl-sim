from abc import ABC, abstractmethod
from typing import Dict, Any

class UpdateProcessor(ABC):
    """
    Handles pre-processing (compression, encryption) and post-processing.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def process_outbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        """Called before broadcasting."""
        return payload

    @abstractmethod
    def process_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Called after receiving."""
        return payload