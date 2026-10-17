from typing import Dict, Any
from .base import ConsensusStrategy

class SimpleConsensus(ConsensusStrategy):
    """A dummy consensus that accepts everything immediately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        return {"payload": payload, "metadata": {"type": "simple", "proposer": node_id}}

    def validate_block(self, block: Dict[str, Any]) -> bool:
        return True