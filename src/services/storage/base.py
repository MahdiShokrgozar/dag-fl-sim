# src/services/storage/base.py

from abc import ABC, abstractmethod
from typing import Any, Tuple

class StorageEngine(ABC):
    """
    Abstract Interface for Off-chain Storage (IPFS/Cloud).
    Responsible for storing model payloads and tracking data size.
    """
    
    @abstractmethod
    def save(self, data: Any) -> Tuple[str, int]:
        """
        Saves data and returns a tuple containing:
        1. Reference ID (CID/Hash)
        2. Size of data in bytes (for traffic metrics)
        """
        pass

    @abstractmethod
    def load(self, ref_id: str) -> Any:
        """
        Retrieves data using the reference ID.
        """
        pass