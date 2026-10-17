# src/utils/utils.py

import logging
import sys
import pickle
import os
import random
import numpy as np
import torch

def setup_logging(loglevel: str, log_file: str = None):
    """
    Configures the logging system.

    Args:
        loglevel (str): Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file (str, optional): File path to save logs. If None, logs only to console.
    """
    # Convert string level to logging constant
    level = getattr(logging, loglevel.upper(), logging.INFO)
    
    # Define standard log format
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates (important for Notebooks/Colab)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # File handler (overwrites previous log with mode='w')
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # console_handler is already attached above, so this reaches the user.
            root_logger.warning(f"Failed to setup log file at {log_file}. Error: {e}")

def get_object_size_in_bytes(obj) -> int:
    """
    Calculates the approximate memory size of a Python object in bytes.
    Uses pickle serialization to estimate network transmission size.
    
    Args:
        obj: Any Python object (dict, list, tensor, etc.)
        
    Returns:
        int: Size in bytes.
    """
    try:
        return len(pickle.dumps(obj))
    except Exception:
        # Fallback for non-picklable objects (less accurate)
        return sys.getsizeof(obj)

def resolve_device(requested: str) -> str:
    """
    Validates a requested compute device before anything downstream trusts it.

    torch.cuda.is_available() returns True even when the installed PyTorch
    build has no compiled kernels for the assigned GPU's compute capability
    (seen on Kaggle: "CUDA error: no kernel image is available for execution
    on the device"). That failure mode doesn't raise here - it raises the
    first time a real op touches the device, deep inside per-node model
    caches and training loops, where every call site catches broad
    exceptions and logs-and-continues. The net effect is a run that finishes
    "successfully" with 0% accuracy and 0 CPU utilization instead of failing
    loudly. Running one real op against the device up front, before any node
    or model cache exists, converts that into a single loud warning and a
    global CPU fallback for the whole run.
    """
    logger = logging.getLogger("DeviceCheck")

    if "cuda" not in requested:
        return requested

    if not torch.cuda.is_available():
        logger.warning(f"⚠️ Requested device '{requested}' but CUDA is not available. Falling back to CPU.")
        return "cpu"

    try:
        (torch.zeros(1, device="cuda") + 1).cpu()
    except Exception as e:
        name = None
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        logger.error(
            f"⚠️ CUDA reports available (device: {name}) but a real op failed: {e}. "
            "This run will use CPU instead of silently producing zero-accuracy results."
        )
        return "cpu"

    return requested

def set_seed(seed: int, device: str = "cpu"):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value.
        device (str): 'cpu' or 'cuda'.
    """
    if seed is None:
        return

    # Must be set before the CUDA context initializes (any torch.cuda.* call
    # below does that) - required for deterministic cuBLAS GEMM ops (Linear
    # layers). No-op if a CUDA context somehow already exists in this
    # process; in that case set it as a real environment variable before
    # launching the process instead.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Ensure deterministic behavior (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.deterministic only covers convolutions; without this, cuBLAS
        # matmuls (Linear layers) and some backward paths can still vary
        # run-to-run on GPU even with every RNG seeded. warn_only=True so an
        # op without a deterministic implementation logs a warning instead
        # of crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)

    logging.info(f"Random seed set to: {seed}")
    

