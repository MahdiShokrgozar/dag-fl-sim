from typing import Dict, List, Any
from .base import BatchingPolicy

class NoBatchPolicy(BatchingPolicy):
    """Trigger transaction immediately for every single update."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def should_batch(self, pending_updates: List[Any], current_time: float) -> bool:
        return len(pending_updates) > 0