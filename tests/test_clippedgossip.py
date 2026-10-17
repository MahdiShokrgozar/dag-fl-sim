"""
ClippedGossip baseline (reviewer item M1; thesis-hub/docs/decisions.md
2026-09-30): update rule and adaptive clipping radius of He, Karimireddy &
Jaggi (arXiv:2202.01545, Sec. 4.1 and App. D.5, Eq. 10).
"""

import math
import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.clippedgossip_aggregator import (
    ClippedGossipAggregator, adaptive_tau, clipped_gossip)


class MockNode:
    def __init__(self, weights):
        self.node_id = "self"
        self.weights = weights


def _payload(block_id, weights):
    return {"hash": block_id, "proposer": block_id, "timestamp": 0.0,
            "payload": {"weights": weights, "is_full_model": True}, "parents": []}


def test_adaptive_tau_eq10():
    # self + 3 parents, W = 1/4, delta 0.2: closest set with weight <= 0.8 is
    # self (0) and the two closest parents (1, 4); the far one (1e6) is left out.
    tau = adaptive_tau([0.0, 1.0, 4.0, 1e6], [0.25] * 4, 0.2)
    assert math.isclose(tau, math.sqrt(0.25 * (1.0 + 4.0)))


def test_no_clipping_equals_gossip_average():
    # Parents inside the radius: plain uniform average of self + parents.
    # (Eq. 10 gives tau = sqrt(W * sum d^2) < d for equidistant parents, so
    # use a parent set where the closest-set radius covers the others.)
    c = torch.zeros(1)
    ps = [torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([0.1])]
    out, tau, n_clipped = clipped_gossip(c, ps, delta=0.0)
    assert math.isclose(tau, math.sqrt(0.75 * 0.01), rel_tol=1e-6)  # 0.0866 < 0.1: clipped
    assert n_clipped == 3
    out, tau, n_clipped = clipped_gossip(c, [torch.tensor([0.01])] * 3 + [torch.tensor([0.1])], delta=0.0)
    # tau = sqrt(0.2*(3*1e-4 + 1e-2)) = 0.0454: the three close parents pass unclipped
    assert n_clipped == 1


def test_poisoned_parent_is_clipped_to_tau():
    c = torch.zeros(2)
    ps = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1e4, 0.0])]
    out, tau, n_clipped = clipped_gossip(c, ps, delta=0.2)
    tau_expected = math.sqrt(0.25 * 2.0)  # self + two honest parents
    assert math.isclose(tau, tau_expected, rel_tol=1e-6)
    assert n_clipped == 3  # the adaptive radius (0.707) also clips the honest parents (distance 1)
    expected = 0.25 * tau * (ps[0] + ps[1]) + 0.25 * torch.tensor([tau, 0.0])
    assert torch.allclose(out, expected, atol=1e-5)


def test_aggregate_centers_on_own_model_and_keeps_layers():
    own = {"w": torch.zeros(2, 2), "n": torch.tensor(3)}
    parents = [_payload(f"b{i}", {"w": torch.full((2, 2), float(v)), "n": torch.tensor(3)})
               for i, v in enumerate([1.0, 1.0, 1e5])]
    agg = ClippedGossipAggregator({"protocols": {"farabi": {"aggregation": {}}}})
    out = agg.aggregate(parents, node=MockNode(own))
    assert set(out) == {"w", "n"} and out["n"].dtype == own["n"].dtype
    assert out["w"].abs().max().item() < 2.0  # the 1e5 parent cannot drag the result
