import torch
from torch.utils.data import Subset
import logging

class FastTensorDataLoader:
    """
    Hybrid High-Performance DataLoader.
    
    Modes:
    1. GPU-Cached (Fastest): If device is provided AND cache_on_device=True, 
       moves entire shard to GPU upfront. Ideal for small datasets (MNIST/CIFAR).
    2. CPU-Cached (Standard): Keeps data in RAM, moves batches to GPU on-the-fly. 
       Use this for large datasets to avoid CUDA OOM.
    """
    def __init__(
        self, 
        dataset, 
        batch_size: int, 
        shuffle: bool = True, 
        device: torch.device = None,
        cache_on_device: bool = True
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if device is not None:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        else:
            self.device = None
      
        self.logger = logging.getLogger("FastLoader")
        self.cache_on_device = cache_on_device
        
        # 1. Extract Tensor Data
        if isinstance(dataset, Subset):
            parent = dataset.dataset
            indices = dataset.indices
            if hasattr(parent, 'data') and hasattr(parent, 'targets'):
                self.dataset_data = parent.data[indices]
                self.dataset_targets = parent.targets[indices]
            else:
                raise ValueError("Dataset must have .data and .targets")
        elif hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            self.dataset_data = dataset.data
            self.dataset_targets = dataset.targets
        else:
            raise ValueError("Dataset must have .data and .targets")

        # 2. Decide Storage Location
        self.is_on_gpu = False
        
        if self.device is not None and self.device.type != 'cpu' and self.cache_on_device:
            try:
                # Try moving to GPU
                self.dataset_data = self.dataset_data.to(self.device)
                self.dataset_targets = self.dataset_targets.to(self.device)
                self.is_on_gpu = True
                self.logger.info(f"🚀 Data shard moved to GPU {self.device} (VRAM Cached).")
            except RuntimeError as e:
                self.logger.warning(f"⚠️ VRAM full! Falling back to CPU caching. Error: {e}")
                self.is_on_gpu = False
        else:
            self.logger.info("ℹ️ Running in CPU-Cache mode (Dynamic Transfer).")
            
        self.data_len = len(self.dataset_data)
        self.n_batches = (self.data_len + batch_size - 1) // batch_size
        self.indices = None

    def __iter__(self):
        # Generate indices on the same device as data to avoid transfers
        dev = self.device if self.is_on_gpu else 'cpu'
        
        if self.shuffle:
            self.indices = torch.randperm(self.data_len, device=dev)
        else:
            self.indices = None
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= self.data_len:
            raise StopIteration

        end = min(self.cursor + self.batch_size, self.data_len)
        
        # Slicing
        if self.indices is not None:
            batch_indices = self.indices[self.cursor:end]
            batch_x = self.dataset_data[batch_indices]
            batch_y = self.dataset_targets[batch_indices]
        else:
            batch_x = self.dataset_data[self.cursor:end]
            batch_y = self.dataset_targets[self.cursor:end]

        self.cursor = end

        # 3. Dynamic Transfer (If not cached on GPU)
        # CRITICAL FIX: Use self.device here as well
        if not self.is_on_gpu and self.device is not None:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
        return batch_x, batch_y

    def __len__(self):
        return self.n_batches