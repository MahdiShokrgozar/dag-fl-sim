# src/services/storage/hybrid.py

import os
import pickle
import hashlib
import shutil
import logging
import zlib
from typing import Any, Tuple
from collections import OrderedDict
from .base import StorageEngine

class HybridStorageAdapter(StorageEngine):
    """
    Hybrid Storage Engine with Compression & Auto-Pruning.
    
    Strategy:
    1. Hot Data: Kept in RAM (LRU Cache) for instant access.
    2. Cold Data: Saved to Disk (Compressed via zlib).
    3. Overflow Protection: Prunes oldest files when limits are reached.
    """
    def __init__(self, root_dir: str = "storage_data", ram_cache_size: int = 50, max_disk_files: int = 500):
        self.root_dir = root_dir
        self.ram_cache_size = ram_cache_size
        self.max_disk_files = max_disk_files
        self.logger = logging.getLogger("HybridStorage")
        
        # LRU Cache for RAM
        self.cache = OrderedDict()
        
        # Tracker for Disk Files (FIFO for pruning)
        self.disk_files_tracker = []
        
        # Initialize Directory
        if os.path.exists(self.root_dir):
            try:
                shutil.rmtree(self.root_dir)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not clear old storage: {e}")
        
        os.makedirs(self.root_dir, exist_ok=True)
        self.logger.info(f"🛡️ Strict Storage initialized. RAM: {ram_cache_size}, Disk Limit: {max_disk_files}")

    def save(self, data: Any) -> Tuple[str, int]:
        try:
            # 1. Compress & Serialize
            # zlib level 1 is fast and provides decent compression for weights
            serialized = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=1)
            size = len(serialized)
            
            # 2. Generate CID
            cid = hashlib.sha256(serialized).hexdigest()
            
            # 3. Update RAM Cache (Priority)
            self._update_ram_cache(cid, data)

            # 4. Persist to Disk (if new)
            path = os.path.join(self.root_dir, cid)
            if not os.path.exists(path):
                # Check limits before writing
                if len(self.disk_files_tracker) >= self.max_disk_files:
                    self._prune_disk(count=50) # Aggressive prune

                with open(path, "wb") as f:
                    f.write(serialized)
                
                self.disk_files_tracker.append(cid)
                
            return cid, size

        except OSError as e:
            if e.errno == 28: # [Errno 28] No space left on device
                self.logger.warning("💾 DISK FULL! Executing emergency pruning...")
                self._prune_disk(count=200) # Emergency clear
                return cid, size # Return anyway, hope it's in RAM or will be retried
            raise e

    def load(self, cid: str) -> Any:
        # 1. Fast Path: Check RAM
        if cid in self.cache:
            self.cache.move_to_end(cid)
            return self.cache[cid]
        
        # 2. Slow Path: Check Disk
        path = os.path.join(self.root_dir, cid)
        try:
            with open(path, "rb") as f:
                compressed_data = f.read()
                data = pickle.loads(zlib.decompress(compressed_data))
                
            # Resurrect to RAM
            self._update_ram_cache(cid, data)
            return data
            
        except FileNotFoundError:
            # Data might have been pruned. This is acceptable in DAG-FL.
            return None
        except Exception as e:
            self.logger.error(f"❌ Error loading CID {cid}: {e}")
            return None

    def _write_to_disk(self, cid: str, compressed_data: bytes):
        """Helper to write data safely."""
        path = os.path.join(self.root_dir, cid)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(compressed_data)
            self.disk_files_tracker.append(cid)

    def _prune_disk(self, count: int):
        """Removes the oldest files from disk to free space."""
        deleted = 0
        while self.disk_files_tracker and deleted < count:
            old_cid = self.disk_files_tracker.pop(0) # Pop from start (Oldest)
            path = os.path.join(self.root_dir, old_cid)
            try:
                if os.path.exists(path):
                    os.remove(path)
                    deleted += 1
            except: pass
        
    def _update_ram_cache(self, cid: str, data: Any):
        """Updates LRU cache in RAM."""
        self.cache[cid] = data
        self.cache.move_to_end(cid)
        if len(self.cache) > self.ram_cache_size:
            self.cache.popitem(last=False) # Remove oldest

    def cleanup(self):
        self.cache.clear()
        if os.path.exists(self.root_dir):
            try: shutil.rmtree(self.root_dir)
            except: pass