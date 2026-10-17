from typing import Dict, List, Any
from .base import BatchingPolicy

class CountBasedBatching(BatchingPolicy):
    """
    Triggers a transaction only when 'N' updates are accumulated.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        batch_config = config.get("strategy_config", {}).get("batching", {})
        if isinstance(batch_config, dict):
             self.threshold = batch_config.get("threshold", 5)
        else:
             self.threshold = 5
        
    def should_batch(self, pending_updates: List[Any], current_time: float) -> bool:
        return len(pending_updates) >= self.threshold