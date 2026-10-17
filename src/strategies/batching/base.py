from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BatchingPolicy(ABC):
    """
    Determines when to trigger a transaction creation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def should_batch(self, pending_updates: List[Any], current_time: float) -> bool:
        """
        Returns True if conditions (size, time, etc.) are met to create a block.
        """
        return len(pending_updates) > 0