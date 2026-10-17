# src/services/storage/ipfs_adapter.py

import torch
import io
import logging
from typing import Dict, Tuple, Any
from .base import StorageEngine

class IPFSStorageAdapter(StorageEngine):
    def __init__(self, config: Dict):
        """
        Initializes IPFS client. Checks for dependency existence at runtime.
        """
        # --- FIX: Lazy Import to prevent crash if library is missing ---
        try:
            global ipfshttpclient
            import ipfshttpclient
        except ImportError:
            logging.error("Dependency 'ipfshttpclient' not found.")
            raise ImportError(
                "You are trying to use IPFS Storage but 'ipfshttpclient' is not installed. "
                "Please run: pip install ipfshttpclient"
            )
        # ---------------------------------------------------------------

        ipfs_conf = config.get("storage", {}).get("ipfs", {})
        
        self.host = ipfs_conf.get("host", "ipfs")
        self.port = ipfs_conf.get("port", 5001)
        self.timeout = ipfs_conf.get("timeout", 30)
        self.client = None
        
        try:
            self._connect()
        except Exception as e:
            logging.warning(f"IPFS init failed (Node might be offline): {e}")

    def _connect(self):
        try:
            addr = f'/dns/{self.host}/tcp/{self.port}/http'
            self.client = ipfshttpclient.connect(
                addr,
                session=True,
                timeout=self.timeout
            )
            logging.info(f"Connected to IPFS node at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"IPFS connection failed: {e}")
            raise e

    def save(self, data: Any) -> Tuple[str, int]:
        if not self.client:
            self._connect()
            
        try:
            buffer = io.BytesIO()
            if hasattr(data, 'state_dict'):
                torch.save(data.state_dict(), buffer)
            else:
                torch.save(data, buffer)
            
            buffer.seek(0)
            data_bytes = buffer.getvalue()
            size = len(data_bytes)
            
            res = self.client.add(buffer, pin=True)
            cid = res['Hash']
            
            return cid, size
            
        except Exception as e:
            logging.error(f"IPFS save failed: {e}")
            return "", 0

    def load(self, cid: str) -> Any:
        if not self.client:
            self._connect()
            
        try:
            res = self.client.cat(cid, timeout=self.timeout)
            buffer = io.BytesIO(res)
            return torch.load(buffer)
        except Exception as e:
            logging.error(f"IPFS load failed for {cid}: {e}")
            return None

    def __del__(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass