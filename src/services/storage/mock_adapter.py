# src/services/storage/mock_adapter.py

import hashlib
import io
import logging
import time
import torch
import pickle
from typing import Dict, Tuple, Any
from .base import StorageEngine  

logger = logging.getLogger(__name__)

class MockStorageAdapter(StorageEngine): 
    """
    In-memory storage simulator that mimics IPFS/Local behavior.
    """
    def __init__(self, config: Dict):
        storage_conf = config.get("storage", {}).get("mock", {})
        self.latency = storage_conf.get("latency", 0.0)
        
        self.storage: Dict[str, bytes] = {}
        logger.info(f"MockStorageAdapter initialized (latency={self.latency}s).")

    def save(self, data: Any) -> Tuple[str, int]: 
        """
        Stores model and returns (CID, Size_in_Bytes).
        """
        # Simulate network latency
        if self.latency > 0:
            time.sleep(self.latency)

        buffer = io.BytesIO()
        
        try:
            if hasattr(data, 'state_dict'):
                torch.save(data.state_dict(), buffer)
            else:
                torch.save(data, buffer)
        except Exception:
            pickle.dump(data, buffer)
            
        data_bytes = buffer.getvalue()
        data_size = len(data_bytes)
        
        # Generate CID similar to IPFS
        cid = hashlib.sha256(data_bytes).hexdigest()
        
        # Store in memory
        self.storage[cid] = data_bytes
        
        return cid, data_size

    def load(self, ref_id: str) -> Any: 
        """
        Retrieves data using CID.
        """
        data_bytes = self.storage.get(ref_id)
        if data_bytes is None:
            logger.error(f"MockStorage: CID {ref_id} not found!")
            return None
            
        buffer = io.BytesIO(data_bytes)
        try:
            return torch.load(buffer)
        except RuntimeError:
            buffer.seek(0)
            return pickle.load(buffer)
        except Exception as e:
            logger.error(f"MockStorage: Failed to load data. {e}")
            return None