"""
Reviewer items M3/M5 (thesis-hub/docs/decisions.md, 2026-09-29):
- when every parent's weight underflows to 0.0 the default still averages the
  parents uniformly (the evaluated rule); zero_weight_fallback: own keeps the
  node's own model instead;
- reputation_weight: false replaces every proposer's Virtue by one constant.
"""

import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.farabi_global_prox_aggregator import FarabiGlobalProxAggregator


class MockReputation:
    def __init__(self, scores):
        self.scores = scores

    def get_virtue_score(self, node_id):
        return self.scores.get(node_id, 50.0)


class MockNode:
    def __init__(self, weights, scores=None):
        self.node_id = "self"
        self.weights = weights
        self.reputation = MockReputation(scores or {})


def _payload(block_id, proposer, weights):
    return {"hash": block_id, "proposer": proposer, "timestamp": 0.0,
            "payload": {"weights": weights, "is_full_model": True}, "parents": []}


def _agg(**aggregation):
    cfg = {"protocols": {"farabi": {"aggregation": {"proximal_mu": 0.5, **aggregation}}}}
    return FarabiGlobalProxAggregator(cfg)


def _far_parents():
    return [_payload(f"b{i}", f"p{i}", {"w": torch.full((4,), 1e4 * (i + 1))}) for i in range(3)]


def test_all_zero_weight_default_is_uniform_average():
    # Default = the rule every reported result used.
    own = {"w": torch.zeros(4)}
    out = _agg().aggregate(_far_parents(), node=MockNode(own))
    assert torch.allclose(out["w"], torch.full((4,), 2e4))


def test_all_zero_weight_own_fallback_keeps_own_model():
    own = {"w": torch.zeros(4)}
    out = _agg(zero_weight_fallback="own").aggregate(_far_parents(), node=MockNode(own))
    assert torch.equal(out["w"], own["w"])
    out["w"] += 1.0  # must be a copy, not the node's live tensor
    assert torch.equal(own["w"], torch.zeros(4))


def test_one_near_parent_takes_all_weight():
    own = {"w": torch.zeros(4)}
    near = _payload("b0", "p0", {"w": torch.full((4,), 0.1)})
    far = _payload("b1", "p1", {"w": torch.full((4,), 1e4)})
    out = _agg().aggregate([near, far], node=MockNode(own))
    assert torch.allclose(out["w"], torch.full((4,), 0.1))


def test_reputation_weight_off_ignores_virtue():
    own = {"w": torch.zeros(4)}
    a = _payload("b0", "a", {"w": torch.full((4,), 0.1)})
    b = _payload("b1", "b", {"w": torch.full((4,), -0.1)})
    node = MockNode(own, scores={"a": 90.0, "b": 10.0})
    on = _agg().aggregate([a, b], node=node)["w"]
    off = _agg(reputation_weight=False).aggregate([a, b], node=node)["w"]
    assert torch.allclose(on, torch.full((4,), 0.1 * (90 - 10) / 100), atol=1e-6)
    assert torch.allclose(off, torch.zeros(4), atol=1e-6)


def test_hard_kernel_is_indicator_of_radius():
    # Second Q1 review, item N1: distance_kernel "hard" = 1 inside hard_radius, 0 outside;
    # reputation still weights the parents inside.
    own = {"w": torch.zeros(4)}
    a = _payload("b0", "a", {"w": torch.full((4,), 1.0)})    # d = 2
    b = _payload("b1", "b", {"w": torch.full((4,), -0.5)})   # d = 1
    c = _payload("b2", "c", {"w": torch.full((4,), 3.0)})    # d = 6, outside
    node = MockNode(own, scores={"a": 30.0, "b": 60.0, "c": 90.0})
    out = _agg(distance_kernel="hard", hard_radius=4.0).aggregate([a, b, c], node=node)["w"]
    assert torch.allclose(out, torch.full((4,), (30 * 1.0 + 60 * -0.5) / 90), atol=1e-6)


def test_default_kernel_unchanged():
    own = {"w": torch.zeros(4)}
    a = _payload("b0", "a", {"w": torch.full((4,), 1.0)})
    b = _payload("b1", "b", {"w": torch.full((4,), -0.5)})
    node = MockNode(own)
    default = _agg().aggregate([a, b], node=node)["w"]
    explicit = _agg(distance_kernel="exp").aggregate([a, b], node=node)["w"]
    assert torch.equal(default, explicit)
