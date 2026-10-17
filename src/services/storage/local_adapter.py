# src/services/storage/local_adapter.py

import logging
import uuid
import torch
import os
from pathlib import Path
from typing import Any, Tuple, Dict
from .base import StorageEngine

logger = logging.getLogger(__name__)


class LocalStorageAdapter(StorageEngine):
    """
    Local filesystem implementation of the storage engine.
    Useful for debugging without running a real IPFS node.
    """
    def __init__(self, config: Dict):
        # Extract path from config, default to 'data/offchain_storage'
        storage_conf = config.get("storage", {})
        self.output_dir = Path(storage_conf.get("local_dir", "data/offchain_storage"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: Any) -> Tuple[str, int]:
        """
        Saves data to local disk and returns (CID, size_bytes).
        """
        # Create unique ID
        cid = uuid.uuid4().hex
        file_path = self.output_dir / f"{cid}.pt"
        
        try:
            # Save payload using torch
            torch.save(data, file_path)
            
            # Calculate actual size in bytes for metrics
            size_bytes = os.path.getsize(file_path)
            
            return cid, size_bytes
        except Exception as e:
            logger.error(f"Local Storage Save Error: {e}")
            return "", 0

    def load(self, ref_id: str) -> Any:
        """
        Loads data from local disk.
        """
        file_path = self.output_dir / f"{ref_id}.pt"
        if file_path.exists():
            try:
                # Load with weights_only=False if you trust the source, or True for safety
                # Since this is a simulation, standard load is fine.
                return torch.load(file_path)
            except Exception as e:
                logger.error(f"Local Storage Load Error: {e}")
                return None
        return None