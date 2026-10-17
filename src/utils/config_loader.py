# src/utils/config_loader.py

import yaml
import json
import os
import logging
from typing import Dict, Any

# Initialize logger for this module
logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a configuration file from the specified path.
    Supports .yaml, .yml, and .json formats.

    Args:
        path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(path):
        error_msg = f"Configuration file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Detect file extension
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif ext == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported format '{ext}'. Use .yaml or .json.")
        
        config = config if config else {}
        logger.info(f"Successfully loaded primary config from {path}")

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {path}: {e}")
        raise
    
    # 2. Dynamically load and merge metric presets
    # This prevents the MetricManager from doing local File I/O operations
    presets_path = os.path.join("configs", "metric_presets.yml")
    metric_presets = {}
    
    if os.path.exists(presets_path):
        try:
            with open(presets_path, 'r', encoding='utf-8') as f:
                presets_data = yaml.safe_load(f) or {}
                # Extract the nested 'metric_presets' block if it exists
                metric_presets = presets_data.get("metric_presets", {})
            logger.info(f"Successfully merged metric presets from {presets_path}")
        except Exception as e:
            logger.error(f"Failed to parse metric presets from {presets_path}: {e}")
    else:
        logger.warning(f"Metric presets file not found at {presets_path}. Using empty presets.")

    # 3. Inject presets into the main configuration tree
    config["metric_presets"] = metric_presets

    return config