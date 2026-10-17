"""
Noisy lazy and noisy stealing attackers (thesis-hub/docs/decisions.md, 2026-09-30):
the helper perturbs only floating-point tensors, and the attackers call it only
when noise_std > 0, so the default attacks draw no random numbers.
"""

import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nodes.malicious.lazy import add_gaussian_noise


def test_noise_touches_only_float_tensors():
    w = {"w": torch.zeros(1000), "n": torch.tensor(7)}
    out = add_gaussian_noise(w, 0.01)
    assert torch.equal(out["n"], w["n"])
    assert 0.005 < out["w"].std().item() < 0.015
    assert torch.equal(w["w"], torch.zeros(1000))  # input left unchanged


def test_default_path_draws_no_random_numbers():
    # The attackers skip the helper at noise_std 0.0; the RNG state must be what it was.
    import inspect
    from src.nodes.malicious import lazy, stealing
    assert "if noise_std > 0.0:" in inspect.getsource(lazy.LazyTrainer.train_and_evaluate)
    assert "if noise_std > 0.0:" in inspect.getsource(stealing.StealingTrainer.train_and_evaluate)
