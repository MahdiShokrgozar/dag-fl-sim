"""
protocols.farabi.aggregation.prefilter (thesis-hub/docs/decisions.md,
2026-09-28; roadmap 2.10): unset keeps each arm's default (Farabi-Prox
filters, literature baselines do not); False removes the filter from
Farabi-Prox; True adds it to BALANCE and RFA.
"""

import os
import sys
import types

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.farabi_global_prox_aggregator import FarabiGlobalProxAggregator
from src.strategies.aggregation.rfa_aggregator import RFAAggregator
from src.strategies.aggregation.balance_aggregator import BalanceAggregator


class _Rep:
    def get_virtue_score(self, node_id):
        return 50.0


def _node():
    return types.SimpleNamespace(node_id="self", weights={"w": torch.zeros(100)}, reputation=_Rep(),
                                 staleness_round_period=None, simulator=types.SimpleNamespace(current_time=0.0))


def _payloads():
    # Two honest parents near the node's own model, one 1000x further away:
    # distance ~1 vs ~1000, above the filter's max(2 * median, 50).
    near1, near2, far = torch.full((100,), 0.1), torch.full((100,), -0.1), torch.full((100,), 100.0)
    return [{"hash": f"b{i}", "proposer": f"p{i}", "timestamp": 0.0,
             "payload": {"weights": {"w": w}, "is_full_model": True}, "parents": []}
            for i, w in enumerate((near1, near2, far))]


def _cfg(prefilter):
    agg = {"proximal_mu": 0.5}
    if prefilter is not None:
        agg["prefilter"] = prefilter
    return {"protocols": {"farabi": {"aggregation": agg}}, "participants": {"total": 10}, "simulation": {"max_time": 600.0}}


def _count_calls(agg, node):
    calls = []
    orig = agg._filter_weight_space_anomalies
    def spy(payloads, n):
        calls.append(len(payloads)); return orig(payloads, n)
    agg._filter_weight_space_anomalies = spy
    agg.aggregate(_payloads(), node=node)
    return calls


def test_defaults():
    # Self-anchored Farabi-Prox runs without the filter since 2026-09-28 (plan 56).
    assert _count_calls(FarabiGlobalProxAggregator(_cfg(None)), _node()) == []
    assert _count_calls(FarabiGlobalProxAggregator(_cfg(True)), _node()) == [3]
    assert _count_calls(RFAAggregator(_cfg(None)), _node()) == []
    assert _count_calls(BalanceAggregator(_cfg(None)), _node()) == []


def test_prefilter_false_removes_it_from_farabi_prox():
    assert _count_calls(FarabiGlobalProxAggregator(_cfg(False)), _node()) == []


def test_prefilter_true_adds_it_to_baselines():
    assert _count_calls(RFAAggregator(_cfg(True)), _node()) == [3]
    assert _count_calls(BalanceAggregator(_cfg(True)), _node()) == [3]


def test_rfa_with_prefilter_drops_the_far_model():
    out_off = RFAAggregator(_cfg(None)).aggregate(_payloads(), node=_node())["w"]
    out_on = RFAAggregator(_cfg(True)).aggregate(_payloads(), node=_node())["w"]
    assert out_on.abs().max() < 0.2          # only the two near models remain
    assert not torch.allclose(out_on, out_off)


def test_side_lists_stay_aligned():
    agg = RFAAggregator(_cfg(True))
    models = [p["payload"]["weights"] for p in _payloads()]
    kept, side = agg._prefilter_models(models, _node(), ["a", "b", "c"])
    assert side == ["a", "b"] and len(kept) == 2
