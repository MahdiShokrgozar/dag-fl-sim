from typing import Dict, Any
from .base import UpdateProcessor

class IdentityProcessor(UpdateProcessor):
    """
    Does nothing. Passes data through unchanged.
    Used when no processing is required.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def process_outbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        return payload

    def process_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload