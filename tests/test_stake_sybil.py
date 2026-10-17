"""
Entry reputation "stake_sybil" (thesis-hub/docs/decisions.md, 2026-10-01): honest
nodes draw exactly the stake of "stake" mode, attacker identities enter with
sybil_stake, and handing over the attacker set re-seeds untouched ledger entries.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.reputation.farabi_reputation import FarabiReputation


def _rep(mode, seed=42):
    cfg = {"simulation": {"seed": seed},
           "protocols": {"farabi": {"reputation": {"initial_virtue_mode": mode}}}}
    return FarabiReputation(cfg)


def test_honest_entry_matches_stake_mode():
    a, b = _rep("stake"), _rep("stake_sybil")
    b.set_attacker_ids({"node_3"})
    for n in ("node_0", "node_1", "node_7"):
        assert a.initial_virtue_for(n) == b.initial_virtue_for(n)


def test_attacker_gets_sybil_stake_floor():
    r = _rep("stake_sybil")
    r.set_attacker_ids({"node_3"})
    # stake_min 1, stake_max 100 -> sqrt(0.01) = 0.1 -> 25 + 0.1 * 50 = 30
    assert abs(r.initial_virtue_for("node_3") - 30.0) < 1e-9


def test_set_attacker_ids_reseeds_untouched_entries_only():
    r = _rep("stake_sybil")
    r.get_virtue_score("node_3")          # seeded before the set is known
    r.get_virtue_score("node_1")
    r.ledger["node_1"] += 5.0             # moved: must be kept
    moved = r.set_attacker_ids({"node_3"})
    assert moved == 1
    assert abs(r.ledger["node_3"] - 30.0) < 1e-9
    assert r.set_attacker_ids({"node_1"}) == 0 and r.attacker_ids == {"node_3"}  # first call only


def test_constant_mode_unchanged():
    r = _rep("constant")
    assert r.set_attacker_ids({"node_3"}) == 0
    assert r.initial_virtue_for("node_3") == 50.0
