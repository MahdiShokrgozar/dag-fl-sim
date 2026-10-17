# src/utils/ipfs_utils.py

import hashlib
import io
import logging
import os
from typing import Dict, Optional, Tuple

import requests
import torch

logger = logging.getLogger(__name__)

class MockIPFS:
    """
    In-Memory Mock for IPFS operations.
    Uses Byte-Serialization to accurately simulate network traffic sizes
    and prevent PyTorch pass-by-reference memory bugs.
    """
    _global_storage = {}
    _is_initialized = False
    
    def __init__(self):
        if not MockIPFS._is_initialized:
            logger.info("💡 MockIPFS global RAM-Storage initialized. Ready for P2P simulation.")
            MockIPFS._is_initialized = True

    def add(self, model_weights: dict) -> Tuple[str, int]:
        """
        Simulates model 'upload'. Returns CID and size.
        """
        buffer = io.BytesIO()
        torch.save(model_weights, buffer)
        data_bytes = buffer.getvalue()
        data_size = len(data_bytes)
        cid = hashlib.sha256(data_bytes).hexdigest()
        
        MockIPFS._global_storage[cid] = data_bytes
        logger.debug(f"🚨 [DEBUG-IPFS-UPLOAD] Successfully saved CID: {cid[:12]}... | Total Files in IPFS: {len(MockIPFS._global_storage)}")
        logger.debug(f"[MockIPFS] Stored {data_size} bytes with CID: {cid[:12]}...")
        return cid, data_size

    def get(self, cid: str) -> Tuple[dict, int]:
        """
        Simulates model 'download'. Returns model weights and size.
        """
        logger.debug(f"🚨 [DEBUG-IPFS-REQUEST] Trying to download CID: {cid}")
        data_bytes = MockIPFS._global_storage.get(cid)
        if data_bytes is None:
            available_cids = [c[:12] for c in MockIPFS._global_storage.keys()]
            logger.debug(f"🚨 [DEBUG-IPFS-MISS] ❌ FATAL: CID {cid} NOT FOUND! Available CIDs in RAM: {available_cids}")
            raise FileNotFoundError(f"MockIPFS: CID {cid} not found in global storage.")
        
        data_size = len(data_bytes)
        buffer = io.BytesIO(data_bytes)
        
        model_weights = torch.load(buffer, map_location='cpu')
        
        logger.debug(f"🚨 [DEBUG-IPFS-DOWNLOAD] ✅ Successfully retrieved CID: {cid[:12]}...")
        logger.debug(f"[MockIPFS] Retrieved {data_size} bytes with CID: {cid[:12]}...")
        return model_weights, data_size

# ==============================================================================
# Main Upload/Download Functions (Hybrid Mode)
# ==============================================================================

def upload_model_to_ipfs(model_weights: dict, storage_config: dict, mock_ipfs_instance: Optional[MockIPFS] = None) -> str:
    """
    Uploads model weights to IPFS (Real or Mock).
    """

    if storage_config.get("ipfs_mock_enabled", True):
        ipfs = mock_ipfs_instance if mock_ipfs_instance else MockIPFS()
        cid, _ = ipfs.add(model_weights)
        return cid
    
    # --- Real IPFS Upload (Pinata) ---
    logger.debug("Performing REAL IPFS upload to Pinata...")
    temp_file_path = "temp_model_for_ipfs.pth"
    torch.save(model_weights, temp_file_path)

    pinata_endpoint = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {
        "pinata_api_key": storage_config.get('pinata_api_key'),
        "pinata_secret_api_key": storage_config.get('pinata_secret_api_key'),
    }

    try:
        with open(temp_file_path, "rb") as f:
            response = requests.post(pinata_endpoint, files={"file": ("model.pth", f)}, headers=headers)
        response.raise_for_status()
        cid = response.json()['IpfsHash']
        logger.debug(f"Successfully uploaded model to Pinata. CID: {cid}")
        return cid
    except requests.exceptions.RequestException as e:
        logger.debug(f"IPFS Upload Failed! Error: {e}")
        raise
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def download_model_from_ipfs(cid: str, storage_config: dict, mock_ipfs_instance: Optional[MockIPFS] = None) -> Tuple[dict, int]:
    """
    Downloads model from IPFS (Real or Mock).
    Returns: (Model Weights Dictionary, Downloaded Bytes)
    """
    
    if storage_config.get("ipfs_mock_enabled", True):
        ipfs = mock_ipfs_instance if mock_ipfs_instance else MockIPFS()
        return ipfs.get(cid)

    # --- Real IPFS Download (Gateway) ---
    logger.debug(f"Performing REAL IPFS download for CID: {cid[:12]}...")
    gateway_url = f"https://gateway.pinata.cloud/ipfs/{cid}"
    try:
        response = requests.get(gateway_url, timeout=60)
        response.raise_for_status()
        
        data_bytes = response.content
        data_size = len(data_bytes)

        buffer = io.BytesIO(data_bytes)
        
        model_weights = torch.load(buffer, map_location='cpu')
        
        logger.debug(f"Successfully downloaded {data_size} bytes from Pinata gateway.")
        return model_weights, data_size
    except requests.exceptions.RequestException as req_err:
        logger.debug(f"IPFS Download Failed for CID {cid}! Error: {req_err}")
        raise