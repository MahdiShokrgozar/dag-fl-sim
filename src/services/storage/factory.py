# src/services/storage/factory.py

import logging
from typing import Dict, Any

from .base import StorageEngine
from .mock_adapter import MockStorageAdapter
from .local_adapter import LocalStorageAdapter
from .ipfs_adapter import IPFSStorageAdapter
from .hybrid import HybridStorageAdapter

class StorageFactory:
    """
    Factory to instantiate the appropriate storage backend.
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> StorageEngine:
        logger = logging.getLogger("StorageFactory")
        storage_cfg = config.get("storage", {})
        
        # Determine type (support backward compatibility)
        if storage_cfg.get("use_ipfs", False):
            storage_type = "ipfs"
        elif storage_cfg.get("ipfs_mock_enabled", False):
            storage_type = "mock"
        else:
            storage_type = storage_cfg.get("type", "mock").lower()

        logger.info(f"Initializing Storage Engine: {storage_type}")

        if storage_type == "hybrid":
            # کش سایز ۵۰ برای ۲۰ نود عالی است (تقریباً ۲ دور کامل در رم می‌ماند)
            return HybridStorageAdapter(
                root_dir="temp_storage_data", 
                ram_cache_size=100 
            )
            
        elif storage_type == "ipfs":
            return IPFSStorageAdapter(config)
            
        elif storage_type == "local":
            return LocalStorageAdapter(config)
            
        elif storage_type == "mock":
            return MockStorageAdapter(config)
            
        else:
            logger.warning(f"Unknown storage type '{storage_type}', defaulting to Mock.")
            return MockStorageAdapter(config)