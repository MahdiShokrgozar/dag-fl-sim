# run_experiments.py
import os
import sys
import yaml
import argparse
import copy
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections.abc import Mapping

def deep_merge(base_dict, update_dict):
    """Recursively merges two dictionaries."""
    for k, v in update_dict.items():
        if isinstance(v, Mapping):
            base_dict[k] = deep_merge(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict

def _run_env(jobs):
    """Environment for every run.

    PYTHONHASHSEED is pinned because set iteration order over node ids
    depends on it: with a random hash seed, two runs of the same config and
    seed differed by up to 5.5 accuracy points (thesis-hub/docs/decisions.md,
    2026-09-28); pinned, they are byte-identical. A value set by the caller
    wins. OMP_NUM_THREADS is deliberately NOT lowered for --jobs > 1: the
    CPU thread count changes floating-point reduction order, so runs would
    no longer match their sequential twins byte for byte.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    return env

_PLATFORM = None


def _platform():
    """Hardware/software the numbers depend on. Same code, config and hash
    seed still gave different accuracies on molab and Kaggle (plan 60c vs
    56, thesis-hub/docs/decisions.md 2026-09-29), so record where a run ran."""
    global _PLATFORM
    if _PLATFORM is None:
        import platform
        lines = [f"python={platform.python_version()}", f"cpu_count={os.cpu_count()}"]
        try:
            import torch
            lines.append(f"torch={torch.__version__}")
            lines.append(f"cuda={torch.version.cuda}")
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "<none>"
            lines.append(f"gpu={gpu}")
        except Exception as e:  # never block a run on this
            lines.append(f"torch=<unavailable: {e}>")
        _PLATFORM = lines
    return _PLATFORM


def _run_one(run_dir, config_path, env=None):
    """Run main.py for one config as an isolated OS process (memory-leak
    prevention); on failure, write its stderr to crash_log.txt."""
    # Both variables change the numbers a run produces, and neither is in the
    # config, so record them next to the results.
    stale = os.path.join(run_dir, "crash_log.txt")
    if os.path.exists(stale):
        os.remove(stale)
    src = env if env is not None else os.environ
    with open(os.path.join(run_dir, "run_env.txt"), "w") as f:
        for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "PYTHONPATH"):
            f.write(f"{key}={src.get(key, '<unset>')}\n")
        for line in _platform():
            f.write(line + "\n")
    result = subprocess.run(
        [sys.executable, "main.py", "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        with open(os.path.join(run_dir, "crash_log.txt"), "w") as f:
            f.write(result.stderr)
    else:
        open(os.path.join(run_dir, "run_done.txt"), "w").close()
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Batch Experiment Runner")
    parser.add_argument("--plan", type=str, required=True, help="Path to experiment plan YAML")
    parser.add_argument("--defaults", type=str, default="configs/defaults.yaml", help="Path to default config")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Runs executed at the same time (default 1 = sequential, the "
                             "historical behaviour). Each run is still its own process; >1 only "
                             "fills an otherwise idle GPU/CPU, it does not change any config.")
    args = parser.parse_args()

    # 1. Load Configurations
    if not os.path.exists(args.plan):
        print(f"❌ Error: Plan file '{args.plan}' not found.")
        return

    with open(args.plan, 'r') as f:
        plan = yaml.safe_load(f)
        
    with open(args.defaults, 'r') as f:
        defaults = yaml.safe_load(f)

    global_plan_config = {k: v for k, v in plan.items() if k not in ['name', 'output_root', 'seeds', 'experiments']}
    
    # 2. Setup Batch Environment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = plan.get('output_root', f"results/batch_{timestamp}")
    global_seeds = plan.get('seeds', [42])

    print(f"🚀 --- Robust Experiment Runner ---")
    print(f"📁 Output Directory: {output_root}")
    print(f"🌱 Seeds: {global_seeds}")
    
    tasks = []  # filled only when --jobs > 1
    env = _run_env(args.jobs)
    # Optional measurement probe (thesis-hub/docs/decisions.md, 2026-09-30):
    # `probe: <name>` puts tools/probes/<name> first on PYTHONPATH, so its
    # sitecustomize.py hooks into the unchanged simulator of every run. One
    # probe per plan, because each probe is a module named sitecustomize.
    probe = plan.get("probe")
    if probe:
        probe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "probes", str(probe))
        if not os.path.isfile(os.path.join(probe_dir, "sitecustomize.py")):
            print(f"❌ Error: probe '{probe}' not found in tools/probes/.")
            return
        repo_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = os.pathsep.join(p for p in (probe_dir, repo_root, env.get("PYTHONPATH", "")) if p)
    _platform()  # once, before any worker thread writes run_env.txt
    print(f"🔒 PYTHONHASHSEED={env['PYTHONHASHSEED']}")

    # 3. Iterate Experiments
    for exp_set in plan.get('experiments', []):
        exp_name = exp_set['name']
        base_scenario = exp_set.get('base_config', {})
        
        print(f"\n📦 Collection: {exp_name}")

        for variation in exp_set.get('variations', []):
            tag = variation['tag']
            overrides = variation.get('overrides', {})
            
            print(f"  > Running Scenario: {tag}")

            # Merge configs for this specific variation
            current_config = copy.deepcopy(defaults)
            deep_merge(current_config, global_plan_config)
            deep_merge(current_config, base_scenario)
            deep_merge(current_config, overrides)

            for seed in global_seeds:
                run_dir = os.path.join(output_root, exp_name, tag, f"seed_{seed}")
                
                # --- FAULT TOLERANCE (Resume Capability) ---
                # Check if this experiment already completed successfully
                # Only a run that exited 0 leaves run_done.txt. metrics_summary.csv
                # is not proof: it is appended every reporting interval, so a
                # crashed or killed run (session stopped mid-run, plan 60c
                # lightyear seed 99) leaves a partial one behind.
                if os.path.exists(os.path.join(run_dir, "run_done.txt")):
                    print(f"    - Seed {seed} ... ⏩ SKIPPED (Already completed)")
                    continue

                print(f"    - Seed {seed} ...", end=" ", flush=True)
                
                # Apply runtime overrides for this run
                if 'simulation' not in current_config:
                    current_config['simulation'] = {}
                current_config['simulation']['seed'] = seed
                
                if 'output' not in current_config:
                    current_config['output'] = {}
                current_config['output']['directory'] = run_dir
                os.makedirs(run_dir, exist_ok=True)

                # --- SUBPROCESS ISOLATION (Memory Leak Prevention) ---
                # Write the exact config for this run to a temporary file
                temp_config_path = os.path.join(run_dir, "run_config.yaml")
                with open(temp_config_path, 'w') as f:
                    yaml.dump(current_config, f)

                if args.jobs > 1:
                    print("queued")
                    tasks.append((f"{exp_name}/{tag}/seed_{seed}", run_dir, temp_config_path))
                    continue
                try:
                    ok = _run_one(run_dir, temp_config_path, env)
                    print("✅ OK" if ok else "❌ FAIL")
                except KeyboardInterrupt:
                    print("\n🛑 Runner stopped by user.")
                    return
                except Exception as e:
                    print(f"❌ FAIL (Runner Error: {e})")

    if tasks:
        print(f"\n⚡ Running {len(tasks)} queued runs, {args.jobs} at a time")

        def worker(task):
            label, run_dir, cfg_path = task
            try:
                ok = _run_one(run_dir, cfg_path, env)
            except Exception as e:
                ok = False
                print(f"    - {label} ... ❌ FAIL (Runner Error: {e})", flush=True)
                return
            print(f"    - {label} ... {'✅ OK' if ok else '❌ FAIL'}", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(worker, tasks))
        except KeyboardInterrupt:
            print("\n🛑 Runner stopped by user.")
            return

    print(f"\n🎉 All tasks finished. Results saved to {output_root}")

if __name__ == "__main__":
    main()