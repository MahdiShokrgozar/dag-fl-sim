# main.py
import argparse
import time
import logging
import sys
import os
from pathlib import Path

# Ensure src is visible
sys.path.append(os.getcwd())

from src.utils.config_loader import load_config
from src.utils.utils import setup_logging, set_seed, resolve_device
from src.metrics.manager import MetricManager
from src.simulators.factory import SimulatorFactory

def parse_args():
    parser = argparse.ArgumentParser(description="DAG-based Federated Learning Simulator")
    
    #main arguments
    parser.add_argument("--config", type=str, default="configs/defaults.yaml", help="Path to YAML config file")
    
    # Override arguments - for quick scenario runs
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override compute device (cpu/cuda)")
    parser.add_argument("--output", type=str, help="Override output directory")
    
    return parser.parse_args()

def override_config(config, args):
    """
    Apply command-line overrides to the configuration dictionary.
    """
    if args.seed is not None:
        config['simulation']['seed'] = args.seed
        print(f"⚠️ Config Override: Seed set to {args.seed}")
        
    if args.device is not None:
        config['simulation']['device'] = args.device
        
    if args.output is not None:
        config['output']['directory'] = args.output
        
    return config

def _write_run_resources(log_dir, wall_s, simulator, device):
    """Wall time, peak memory and the run_timing breakdown of this run (tool paper E1)."""
    import json
    import resource
    from src.utils.run_timing import TOTALS, COUNTS
    out = {"wall_s": round(wall_s, 3),
           "sim_time": float(getattr(simulator, "current_time", 0.0)),
           "num_nodes": len(getattr(simulator, "nodes", {}) or {}),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
           "seconds": {k: round(v, 3) for k, v in TOTALS.items()},
           "calls": dict(COUNTS)}
    try:
        import torch
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            out["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        with open(os.path.join(log_dir, "run_resources.json"), "w") as f:
            json.dump(out, f, indent=1)
    except Exception:
        pass


def main():
    # 1. Parse CLI Arguments
    args = parse_args()

    # 2. Load Configuration
    try:
        config = load_config(args.config)
        config = override_config(config, args) 
    except Exception as e:
        print(f"Critical Error loading config: {e}")
        return

    # 3. Setup Logging
    # save logs to output directory
    log_dir = config.get("output", {}).get("directory", "output")
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(config.get("loglevel", "INFO"), log_file=os.path.join(log_dir, "simulation.log"))
    
    logger = logging.getLogger("Main")
    logger.info(f"Starting experiment with config: {args.config}")

    # 4. Set Random Seed (Critical for Reproducibility)
    
    seed = config.get("simulation", {}).get("seed", 42)
    device = resolve_device(config.get("simulation", {}).get("device", "cpu"))
    config["simulation"]["device"] = device
    set_seed(seed, device)
    
    # 5. Initialize Metrics (Observer)
    metric_manager = MetricManager(config)

    # 6. Initialize Simulator (Engine)
    try:
        simulator = SimulatorFactory.get_simulator(config, metric_manager)
    except Exception as e:
        logger.exception(f"Factory failed to create simulator: {e}")
        sys.exit(1)
        # return

    # 7. Run Simulation
    # A crash must end with a non-zero exit code. It used to be logged and then
    # swallowed (exit 0), so run_experiments.py counted a crashed run as OK and
    # wrote no crash_log.txt: plan 59 lost a run to a dataset-download race this
    # way, visible only as an empty metrics file (thesis-hub decisions 2026-09-29).
    crashed = False
    t_run = time.perf_counter()
    try:
        simulator.run()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        crashed = True
    except Exception as e:
        logger.exception(f"Simulation crashed: {e}")
        crashed = True
    finally:
        if metric_manager:
            metric_manager.finalize()
        _write_run_resources(log_dir, time.perf_counter() - t_run, simulator, device)
    if crashed:
        sys.exit(1)

if __name__ == "__main__":
    main()