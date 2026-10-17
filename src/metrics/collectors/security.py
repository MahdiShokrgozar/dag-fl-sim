"""
Security and Threat Detection Metric Collector.

This module evaluates the resilience of the consensus and reputation protocols
against adversarial actions (e.g., model stealing, lazy nodes, data poisoning).
It tracks individual node penalizations and calculates system-wide detection
efficacy using recall (TPR), false positive rate (FPR) and precision.

Why strikes are reported alongside quarantine
---------------------------------------------
Penalisation is graduated: `FarabiReputation.slash_node` increments a strike and
only quarantines once `max_strikes` is reached. Reporting quarantine alone hides
every node that was penalised but not yet quarantined. That is not hypothetical —
in plan 22 group `1_honest_cost/ufs_gates` both wrongly penalised honest nodes
(node_1, node_7) sat at exactly 1 strike out of 2, so this collector reported
`false_positive_rate_pct = 0.0` for a run whose log contains two `L1 SLASH`
lines. Strikes are the leading indicator and must be visible.

Why precision is reported alongside recall
------------------------------------------
Recall on its own rewards indiscriminate slashing. In plan 22
`2_under_attack/poison_ufs_gates` both attackers were caught (recall 100%) while
all 12 nodes were struck (strike precision 23.1%). A table carrying only recall
would have called that run a success.

See thesis-hub/docs/decisions.md (2026-09-06, "باگ متریک") for the full finding.
"""

import math
from typing import List, Dict, Any, Set, Tuple
from ..interface import IMetric


class SecurityMetric(IMetric):
    """
    Measures the efficacy of the slashing and reputation mechanisms.

    Penalisation state is inherently a per-observer view in these protocols:
    every node keeps its own strike ledger and quarantine set. A node counts
    as penalised for reporting purposes once enough peers agree, which is
    what `_peer_votes` computes.
    """

    #: Minimum number of distinct peers that must have independently reached
    #: local quarantine before a node counts as network-detected/slashed here.
    #:
    #: Until 2026-09-09 this was a FRACTION (`QUORUM_RATIO = 0.5`, i.e. 50% of
    #: ALL peers), on top of each peer's own quarantine already requiring
    #: `max_strikes` (2-3) separate confirmed slash events. In a real run
    #: (group 3, mixed_attack, 2026-09-09) that combination reported 0%
    #: detection AND 0% false-positive for an entire 400s run despite real,
    #: confirmed `slash_node` calls (see farabi_dag.py) against every one of
    #: the four actually-malicious nodes, and against several honest ones -
    #: the bar was so high nothing ever crossed it, in either direction. As of
    #: the same day, a peer's own quarantine already requires genuine
    #: evidence-quorum (majority network stake + >=3 distinct approvers,
    #: `_evidence_meets_quorum` in farabi_dag.py) repeated `max_strikes`
    #: times - so ONE peer reaching quarantine is no longer "one node's raw
    #: opinion," it is already a quorum-backed outcome observed from one
    #: vantage point. Requiring a further 50%-of-all-peers layer on top was
    #: redundant double-quorum, not extra safety. See
    #: thesis-hub/docs/decisions.md, 2026-09-09.
    #:
    #: Caveat NOT addressed by this change: a separate, still-unreviewed
    #: strike path (`L1 BATCH SLASH` in farabi_dag.py, majority-of-own-
    #: observed-block-rejections) has no quorum check at all, so a peer's
    #: `quarantine_votes` here could in principle still include contamination
    #: from that path - tracked as an open item, not solved here.
    MIN_DETECTING_PEERS = 1

    @property
    def name(self) -> str:
        """Unique identifier for the metric collector."""
        return "security"

    def get_global_columns(self) -> List[str]:
        """Defines the CSV columns for global security evaluation."""
        return [
            "malicious_nodes_count",
            "slashed_nodes_count",
            "slashed_malicious_count",
            "slashed_honest_count",
            "struck_nodes_count",
            "struck_malicious_count",
            "struck_honest_count",
            "detection_rate_pct",
            "false_positive_rate_pct",
            "slashing_precision_pct",
            "strike_precision_pct",
            # Operational containment, reported alongside the ledger columns
            # above rather than replacing them - see `_is_locally_blocked`.
            "blocked_nodes_count",
            "blocked_malicious_count",
            "blocked_honest_count",
            "block_detection_rate_pct",
            "block_false_positive_rate_pct",
            "block_precision_pct",
        ]

    def get_node_columns(self) -> List[str]:
        """Defines the CSV columns for tracking node-specific security states."""
        return [
            "node_type",
            "is_malicious",
            "reputation_score",
            "is_slashed",
            "is_locally_blocked",
            "strike_votes",
            "strike_total",
            "quarantine_votes",
        ]

    @staticmethod
    def _peer_votes(node: Any) -> Tuple[int, int, int, int]:
        """
        Poll every peer's reputation ledger for its view of `node`.

        Returns (peer_count, strike_votes, strike_total, quarantine_votes):
          - strike_votes: peers holding at least one strike against this node
          - strike_total: sum of strikes across all peers (event volume)
          - quarantine_votes: peers that have quarantined this node

        A node cannot report this about itself — the strike ledgers live on the
        *observers* — so this reaches the peer set through the simulator, the
        same route `farabi_dag` already uses for settlement.
        """
        simulator = getattr(node, "simulator", None)
        peers = getattr(simulator, "nodes", None) or {}
        node_id = str(getattr(node, "node_id", ""))

        peer_count = 0
        strike_votes = 0
        strike_total = 0
        quarantine_votes = 0

        for peer in peers.values():
            reputation = getattr(peer, "reputation", None)
            if reputation is None:
                continue
            peer_count += 1

            strikes = int(getattr(reputation, "strikes", {}).get(node_id, 0))
            if strikes > 0:
                strike_votes += 1
                strike_total += strikes

            is_quarantined = getattr(reputation, "is_quarantined", None)
            if callable(is_quarantined) and is_quarantined(node_id):
                quarantine_votes += 1

        return peer_count, strike_votes, strike_total, quarantine_votes

    def collect(self, node: Any) -> Dict[str, Any]:
        """
        Extracts the security profile and penalisation standing of a given node.

        Args:
            node (Any): The simulation node instance.

        Returns:
            Dict[str, Any]: Security metrics representing the node's current standing.
        """
        stats: Dict[str, Any] = {}
        if hasattr(node, "snapshot_metrics"):
            stats = node.snapshot_metrics()

        rep_score = (
            node.get_current_reputation()
            if hasattr(node, "get_current_reputation")
            else stats.get("reputation_score", 1.0)
        )

        peer_count, strike_votes, strike_total, quarantine_votes = self._peer_votes(node)

        # `node.is_slashed` is only ever written by the IronForge settlement path
        # (ironforge_settlement.py). FARABI penalises through the reputation
        # module instead and leaves that attribute at its `False` default, which
        # is why every FARABI run before 2026-09-06 reported zero slashing. Treat
        # the explicit flag as one of two sources, not the only one.
        explicit_flag = bool(stats.get("is_slashed", getattr(node, "is_slashed", False)))
        is_slashed = explicit_flag or (quarantine_votes >= self.MIN_DETECTING_PEERS)

        return {
            "node_type": str(stats.get("node_type", getattr(node, "node_type", "unknown"))),
            "is_malicious": bool(stats.get("is_malicious", getattr(node, "is_malicious", False))),
            "reputation_score": float(rep_score),
            "is_slashed": is_slashed,
            "is_locally_blocked": self._is_locally_blocked(node),
            "strike_votes": strike_votes,
            "strike_total": strike_total,
            "quarantine_votes": quarantine_votes,
        }

    @staticmethod
    def _is_locally_blocked(node: Any) -> bool:
        """
        Is this node operationally EXCLUDED from participating, by whatever
        mechanism the active protocol uses?

        This is a different question from `is_slashed` above, which reads the
        reputation ledger (strikes / quarantine). Exclusion is about whether a
        node's blocks actually stop being accepted and built upon.

        Protocols express it differently, and the metric has to span them or the
        security axis is not comparable across a baseline table:
          - FARABI: membership of `FarabiDAGManager.locally_slashed_nodes`. That
            set is what `broadcast_to_l2` checks (`SLASH-GATE`) and what
            `FarabiSelection._apply_security_filters` uses to drop a proposer
            from both the tip pool and the finalized anchor pool.
          - IronForge: `node.is_slashed`, set during Algorithm 4 settlement once
            a balance falls to `slashing_threshold`. IronForge has no quarantine
            set at all; its incentive layer is the balance ledger.
        The two are disjoint in practice - FARABI never writes `is_slashed`, and
        IronForge has no `locally_slashed_nodes` - so the union below double
        counts nothing.

        Reading only ONE of them is how this went wrong twice already. Reporting
        only the ledger made the table wrong in both directions on 2026-09-12
        (a `dual` run reporting 75% detection had excluded nobody; a `unified`
        run reporting 0% had excluded all four attackers). Reporting only
        `locally_slashed_nodes` then made every non-FARABI cell in plan 24 read
        0.0% - not because those protocols caught nothing, but because their
        exclusions live somewhere this function was not looking.

        Naming note: the column is still `is_locally_blocked` for continuity
        with plan 24's read-off rules and the 2026-09-12 log entries, even
        though the predicate is now protocol-agnostic. Treat it as "excluded".
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        node_id = str(getattr(node, "node_id", ""))

        # FARABI-style: a shared blacklist held on the L2 DAG manager.
        l2 = getattr(node, "l2_strategy", None)
        blocked = getattr(l2, "locally_slashed_nodes", None)
        if isinstance(blocked, (set, frozenset)) and node_id in blocked:
            return True

        # IronForge-style: a per-node flag written by settlement.
        if bool(getattr(node, "is_slashed", False)):
            return True

        # Some settlement paths also mark the role on the node's L2 state.
        l2_state = getattr(node, "l2_state", None)
        if isinstance(l2_state, dict) and l2_state.get("role") == "slashed":
            return True

        return False

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        """
        Calculates network-wide security performance matrices.

        Recall, FPR and precision are all reported because no one of them is
        sufficient on its own: recall alone rewards slashing everybody, and FPR
        alone rewards slashing nobody.

        Args:
            results (List[Dict[str, Any]]): Collected metrics from all nodes.
            global_state (Any, optional): Global simulator state reference.

        Returns:
            Dict[str, Any]: Global security summary.
        """
        if not results:
            return {}

        malicious_ids: Set[str] = {
            str(r.get("node_id")) for r in results if r.get("is_malicious") is True
        }
        honest_ids: Set[str] = {
            str(r.get("node_id")) for r in results if r.get("is_malicious") is False
        }
        slashed_ids: Set[str] = {
            str(r.get("node_id")) for r in results if r.get("is_slashed") is True
        }
        struck_ids: Set[str] = {
            str(r.get("node_id")) for r in results if int(r.get("strike_votes", 0) or 0) > 0
        }

        blocked_ids: Set[str] = {
            str(r.get("node_id")) for r in results if r.get("is_locally_blocked") is True
        }

        slashed_malicious = len(slashed_ids & malicious_ids)
        slashed_honest = len(slashed_ids & honest_ids)
        struck_malicious = len(struck_ids & malicious_ids)
        struck_honest = len(struck_ids & honest_ids)
        blocked_malicious = len(blocked_ids & malicious_ids)
        blocked_honest = len(blocked_ids & honest_ids)

        def _ratio(numerator: int, denominator: int) -> float:
            """Percentage, or NaN when the denominator makes the question meaningless."""
            if denominator <= 0:
                return math.nan
            return float(round(numerator / denominator * 100.0, 2))

        # NaN rather than 100.0 when there are no adversaries: an all-honest run
        # has no detection rate to report, and the old `else 100.0` fallback made
        # all three arms of plan 22 group 1 read as "100% detection".
        return {
            "malicious_nodes_count": len(malicious_ids),
            "slashed_nodes_count": len(slashed_ids),
            "slashed_malicious_count": slashed_malicious,
            "slashed_honest_count": slashed_honest,
            "struck_nodes_count": len(struck_ids),
            "struck_malicious_count": struck_malicious,
            "struck_honest_count": struck_honest,
            "detection_rate_pct": _ratio(slashed_malicious, len(malicious_ids)),
            "false_positive_rate_pct": _ratio(slashed_honest, len(honest_ids)),
            "slashing_precision_pct": _ratio(slashed_malicious, len(slashed_ids)),
            "strike_precision_pct": _ratio(struck_malicious, len(struck_ids)),
            "blocked_nodes_count": len(blocked_ids),
            "blocked_malicious_count": blocked_malicious,
            "blocked_honest_count": blocked_honest,
            "block_detection_rate_pct": _ratio(blocked_malicious, len(malicious_ids)),
            "block_false_positive_rate_pct": _ratio(blocked_honest, len(honest_ids)),
            "block_precision_pct": _ratio(blocked_malicious, len(blocked_ids)),
        }
