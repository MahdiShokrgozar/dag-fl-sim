"""
Architecture-paper flags (thesis-hub/docs/decisions.md, 2026-10-01), all off by
default: DATS distance only for the sampled candidates, the traffic-scaled finality
threshold of the design document, and random pool assignment.
"""

import os
import sys
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.selection.farabi_selection import FarabiTipSelector
from src.strategies.l2.farabi_dag import FarabiDAGManager


def _cfg(selection=None, l2=None, seed=42):
    return {"simulation": {"seed": seed},
            "protocols": {"farabi": {"selection": selection or {}, "l2": l2 or {}}}}


# ---------------------------------------------------------------- DATS candidates
def _select(scope, n_tips=40):
    sel = FarabiTipSelector(_cfg(selection={"l2_distance_scope": scope, "num_tips": 3,
                                            "candidate_pool_size": 10}))
    tips = [{"hash": f"h{i}", "proposer": f"node_{i % 9 + 1}", "timestamp": 0.0} for i in range(n_tips)]
    sel._get_filtered_tips = lambda dag, node: tips
    calls = []
    def fake_mult(tip, node, exponent):
        calls.append(tip["hash"])
        return 1.0
    sel._l2_trust_multiplier = fake_mult
    node = SimpleNamespace(node_id="node_0", node_accuracy=50.0, per_class_accuracy={})
    chosen, _ = sel.select_tips_for_aggregation(dag=None, node=node, current_time=10.0)
    return chosen, calls


def test_scope_all_computes_distance_for_every_tip():
    chosen, calls = _select("all")
    assert len(calls) == 40 and len(chosen) == 3


def test_scope_candidates_computes_distance_only_for_sampled():
    chosen, calls = _select("candidates")
    assert len(calls) == 10 and len(chosen) == 3
    assert set(chosen) <= set(calls)


# ---------------------------------------------------------------- finality
def test_local_block_rate_window():
    m = FarabiDAGManager(_cfg(l2={"finality_rule": "traffic", "finality_rate_window": 10.0}))
    for t in range(0, 30):          # one block per second for 30 s
        m._arrival_times.append(float(t))
        m._first_arrival = 0.0 if m._first_arrival is None else m._first_arrival
    assert abs(m._local_block_rate(29.5) - 1.0) < 0.15


def test_default_rule_unchanged_and_traffic_threshold():
    assert FarabiDAGManager(_cfg()).finality_rule == "mean_virtue"
    m = FarabiDAGManager(_cfg(l2={"finality_rule": "traffic", "finality_target_time": 2.0, "theta": 0.5, "w_min": 0.0}))
    rep = SimpleNamespace(get_average_network_virtue=lambda: 50.0)
    m.node = SimpleNamespace(reputation=rep, simulator=SimpleNamespace(current_time=20.0))
    m._first_arrival = 0.0
    m._arrival_times.extend([float(t) for t in range(20)])   # 1 block/s
    m.blocks["b"] = {"timestamp": 5.0}
    m.approval_weights["b"] = 0.0
    m.finalized_blocks = set()
    finalized = []
    m._trigger_virtual_milestone = lambda bid, w, th: finalized.append(th)
    # threshold = 0.5 * 50 * 1.0 * 2.0 = 50
    m.update_approval_weights("b", 30.0, approver_id="node_1")
    assert not finalized
    m.update_approval_weights("b", 30.0, approver_id="node_2")
    assert finalized and abs(finalized[0] - 50.0) < 3.0


# ---------------------------------------------------------------- pools
def test_pool_order_contiguous_default_and_random_permutation():
    ids = sorted(f"node_{i}" for i in range(20))
    m = FarabiDAGManager(_cfg())
    assert m.pool_order(ids, 0.0) == ids
    r = FarabiDAGManager(_cfg(l2={"pool_assignment": "random"}, seed=42))
    o1 = r.pool_order(ids, 0.0)
    assert sorted(o1) == ids and o1 != ids
    assert r.pool_order(ids, 999.0) == o1                      # one draw per run
    assert FarabiDAGManager(_cfg(l2={"pool_assignment": "random"}, seed=7)).pool_order(ids, 0.0) != o1
    e = FarabiDAGManager(_cfg(l2={"pool_assignment": "random", "pool_epoch_seconds": 100}, seed=42))
    assert e.pool_order(ids, 50.0) != e.pool_order(ids, 150.0)
