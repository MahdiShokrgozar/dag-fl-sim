"""
In-band attack modes of PoisoningTrainer (thesis-hub/docs/decisions.md,
2026-09-29; Q1 reviewer item M2): ALIE publishes mu - z*sigma of the honest
models (z from Baruch et al.), IPM moves the previous honest mean against the
honest update. The default mode must stay the original Gaussian attack.
"""

import math
import os
import sys
import types

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nodes.malicious.poisoning import PoisoningTrainer


def _node(n_honest, n_poison, cfg, honest_vals):
    nodes = {}
    for i in range(n_honest):
        nodes[f"h{i}"] = types.SimpleNamespace(node_type="honest", weights={"w": torch.full((3,), honest_vals[i])})
    for i in range(n_poison):
        nodes[f"p{i}"] = types.SimpleNamespace(node_type="poisoning", weights={"w": torch.zeros(3)})
    sim = types.SimpleNamespace(nodes=nodes)
    node = types.SimpleNamespace(node_id="p0", config={"participants": {"attacks": {"poisoning": cfg}}},
                                 simulator=sim, logger=types.SimpleNamespace(info=lambda *a, **k: None))
    return node


def _trainer(node):
    t = PoisoningTrainer.__new__(PoisoningTrainer)
    t.node = node
    return t


def test_alie_baruch_z_for_10_nodes_2_attackers():
    node = _node(8, 2, {"mode": "alie"}, [1, 2, 3, 4, 5, 6, 7, 8])
    t = _trainer(node)
    assert abs(t._alie_z({}) - 0.2533) < 1e-3
    out = t._in_band_weights({"w": torch.zeros(3)}, "alie", {"mode": "alie"})
    vals = torch.tensor([1., 2, 3, 4, 5, 6, 7, 8])
    expected = vals.mean() - t._alie_z({}) * vals.std()
    assert torch.allclose(out["w"], torch.full((3,), expected.item()), atol=1e-5)


def test_alie_explicit_z():
    node = _node(4, 1, {"mode": "alie", "alie_z": 1.0}, [0, 2, 4, 6])
    out = _trainer(node)._in_band_weights({"w": torch.zeros(3)}, "alie", {"alie_z": 1.0})
    vals = torch.tensor([0., 2, 4, 6])
    assert torch.allclose(out["w"], torch.full((3,), (vals.mean() - vals.std()).item()), atol=1e-5)


def test_ipm_moves_previous_mean_against_the_honest_update():
    node = _node(2, 1, {"mode": "ipm", "ipm_eps": 0.5}, [1, 3])
    t = _trainer(node)
    first = t._in_band_weights({"w": torch.zeros(3)}, "ipm", {"ipm_eps": 0.5})
    assert torch.allclose(first["w"], torch.full((3,), 2.0))          # no history: mu
    for n in node.simulator.nodes.values():                            # honest mean moves 2 -> 4
        if n.node_type == "honest":
            n.weights = {"w": n.weights["w"] + 2.0}
    second = t._in_band_weights({"w": torch.zeros(3)}, "ipm", {"ipm_eps": 0.5})
    assert torch.allclose(second["w"], torch.full((3,), 2.0 - 0.5 * 2.0))  # mu_prev - eps*(mu - mu_prev)


def test_no_honest_model_returns_none():
    node = _node(0, 2, {"mode": "alie"}, [])
    assert _trainer(node)._in_band_weights({"w": torch.zeros(3)}, "alie", {}) is None
