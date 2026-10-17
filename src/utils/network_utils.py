import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_network_payload_size(payload: Dict[str, Any]) -> int:
    """
    Calculates the exact byte size of a dictionary payload by simulating 
    JSON serialization. This provides a standardized, unbiased metric 
    for evaluating communication overhead across different protocols.
    
    Args:
        payload (Dict[str, Any]): The data payload being transmitted over the network.
        
    Returns:
        int: The size of the serialized payload in bytes. Returns 0 if serialization fails.
    """
    if not payload:
        return 0
        
    try:
        # sort_keys=True ensures consistent byte sizing for identical dictionaries
        serialized_payload = json.dumps(payload, sort_keys=True).encode('utf-8')
        return len(serialized_payload)
    except TypeError as e:
        logger.error(f"Failed to serialize network payload for size calculation: {e}")
        # In a strict academic simulation, we should not ignore non-serializable data.
        # Ensure your L1 and L2 payloads contain only standard data types (str, int, float, list, dict).
        return 0