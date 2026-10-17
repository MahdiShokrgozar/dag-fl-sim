"""
Staleness weight in FarabiGlobalProxAggregator (thesis-hub/docs/decisions.md,
2026-09-28): the self-anchored arm must apply the same exp(-lambda * age)
factor as FarabiProxAggregator, and lambda_aggregate = 0 must reproduce the
plan 50/53 behaviour (no aggregation-side staleness).
"""

import math
import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.farabi_global_prox_aggregator import FarabiGlobalProxAggregator

PERIOD = 10.0
NOW = 100.0


class MockReputation:
    def get_virtue_score(self, node_id):
        return 50.0


class MockNode:
    def __init__(self, weights):
        self.node_id = "self"
        self.weights = weights
        self.reputation = MockReputation()
        self.staleness_round_period = PERIOD
        self.simulator = type('Sim', (), {'current_time': NOW})()


def _payload(block_id, proposer, weights, timestamp):
    return {"hash": block_id, "proposer": proposer, "timestamp": timestamp,
            "payload": {"weights": weights, "is_full_model": True}, "parents": []}


def _run(lambda_aggregate, enabled=True):
    cfg = {"protocols": {"farabi": {
        "aggregation": {"proximal_mu": 0.5},
        "selection": {"staleness": {"enabled": enabled, "lambda_aggregate": lambda_aggregate}},
    }}}
    agg = FarabiGlobalProxAggregator(cfg)
    local = {"w": torch.zeros(4)}
    delta = torch.full((4,), 0.1)
    # Equidistant from the anchor, equal reputation: only age separates them.
    fresh = _payload("b1", "a", {"w": local["w"] + delta}, NOW)
    old = _payload("b2", "b", {"w": local["w"] - delta}, NOW - 2 * PERIOD)
    return agg.aggregate([fresh, old], node=MockNode(local))["w"], delta


def test_old_parent_counts_for_less():
    out, delta = _run(lambda_aggregate=1.0)
    w_fresh, w_old = 1.0, math.exp(-2.0)
    expected = delta * (w_fresh - w_old) / (w_fresh + w_old)
    assert torch.allclose(out, expected, atol=1e-5)


def test_lambda_zero_matches_plan50_behaviour():
    out, _ = _run(lambda_aggregate=0.0)
    assert torch.allclose(out, torch.zeros(4), atol=1e-6)


def test_disabled_matches_plan50_behaviour():
    out, _ = _run(lambda_aggregate=1.0, enabled=False)
    assert torch.allclose(out, torch.zeros(4), atol=1e-6)
