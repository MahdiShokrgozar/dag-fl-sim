import json
import logging
import zlib
import base64
from typing import Dict, Any
from .base import UpdateProcessor

logger = logging.getLogger(__name__)


class GzipProcessor(UpdateProcessor):
    """
    Compresses the payload using Gzip (zlib) to simulate bandwidth savings.
    """
    def __init__(self, config: Dict[str, Any], level: int = 6):
        super().__init__(config)
        self.level = level

    def process_outbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        # 1. Convert the dictionary to a JSON string
        json_str = json.dumps(payload, default=str) # default=str to handle special types
        
        # 2. Compress the JSON string using Gzip
        compressed_data = zlib.compress(json_str.encode('utf-8'), level=self.level)
        
        # 3. Encode the compressed data to base64 for safe JSON transmission
        b64_data = base64.b64encode(compressed_data).decode('utf-8')
        
        return {
            "data": b64_data,
            "is_compressed": True,
            "algo": "gzip"
        }

    def process_inbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        if not payload.get("is_compressed"):
            return payload

        try:
            # 1. Decode the base64 compressed data
            compressed_data = base64.b64decode(payload["data"])
            
            # 2. Decompress the Gzip compressed data
            json_str = zlib.decompress(compressed_data).decode('utf-8')
            
            # 3. Convert the JSON string back to a dictionary
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Decompression Failed: {e}")
            return {}
