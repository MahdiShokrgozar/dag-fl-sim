"""
Module: farabi_reputation.py
Description:
    Implements the global Layer-2 Reputation State (Ledger) for FARABI.
    Enforces the T²RC Game-Theoretic penalty system using a Strike mechanism,
    allowing for graduated slashing, cooling off, and strict quarantine.
"""

import hashlib
import logging
import math
from time import time
from typing import Dict, Any, Set

from src.strategies.reputation.base import BaseReputation

logger = logging.getLogger(__name__)

class FarabiReputation(BaseReputation):
    """
    Manages trust scores (virtue) and strike counts for nodes in the Farabi Protocol.
    Integrates with FarabiConsensus to dynamically reward good behavior or 
    progressively penalize malicious/anomalous actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = "unbound" # Will be set by the node using this module
        # Existing fields
        self.ledger: Dict[str, float] = {}
        self.strikes: Dict[str, int] = {}
        self.quarantined: Set[str] = set()
        
        # New fields for consensus-based quarantine
        self.quarantine_proposals: Dict[str, Dict[str, Any]] = {}
        self.my_votes: Set[str] = set()
        
        farabi_cfg = config.get("protocols", {}).get("farabi", {})
        rep_cfg = farabi_cfg.get("reputation", {})
        consensus_cfg = farabi_cfg.get("consensus", {})
        l2_cfg = farabi_cfg.get("l2", {})

        self.initial_virtue = rep_cfg.get("initial_virtue", 50.0)

        # --- Entry reputation derived from staked collateral -------------------
        # "constant" (default) gives every node the same initial_virtue, which is
        # what every run before 2026-09-05 did. "stake" models the real-world case
        # where a joining node posts collateral to enter an L2 task and receives an
        # entry reputation proportional to it.
        #
        # Two deliberate properties:
        #   1. CONCAVE (sqrt) mapping — capital buys sub-linear trust. This mirrors
        #      the Sybil-resistance already used in tip selection
        #      (farabi_selection.py:264, trust_power = sqrt(virtue/100)).
        #   2. CEILING BELOW THE TIER-3 GATE — stake alone must never purchase
        #      fast-path exemption. The fast path validates only self-reported
        #      metadata (model_fingerprint is not covered by metadata_commitment),
        #      so a node that bought its way into Tier-3 could publish unverified
        #      blocks. Fast-path admission must be EARNED via observed approvals.
        self.initial_virtue_mode = rep_cfg.get("initial_virtue_mode", "constant")
        # "stake_sybil": honest nodes draw their stake exactly as in "stake" (same
        # hash of (seed, node_id), so the two modes are paired per seed); attacker
        # identities enter with `sybil_stake` (default stake_min), the case of an
        # attacker that splits its capital over many identities. Entry reputation is
        # then informative about honesty, which "stake" (random, uncorrelated) is not.
        # The attacker set is handed over by the L2 strategy (set_attacker_ids) at
        # the first topology update, which also re-seeds ledgers created earlier.
        self.stake_min = float(rep_cfg.get("stake_min", 1.0))
        self.sybil_stake = float(rep_cfg.get("sybil_stake", self.stake_min))
        self.attacker_ids: Set[str] = set()
        self.attacker_ids_set = False
        self.stake_max = float(rep_cfg.get("stake_max", 100.0))
        self.stake_virtue_floor = float(rep_cfg.get("stake_virtue_floor", 25.0))
        self.stake_virtue_ceiling = float(rep_cfg.get("stake_virtue_ceiling", 75.0))
        self._stake_seed = config.get("simulation", {}).get("seed", 0)
        self.initial_stake: Dict[str, float] = {}

        tier3 = consensus_cfg.get("tier3_threshold", 0.8) * 100.0
        if self.initial_virtue_mode in ("stake", "stake_sybil") and self.stake_virtue_ceiling >= tier3:
            logger.warning(
                f"⚠️ [STAKE] stake_virtue_ceiling={self.stake_virtue_ceiling} >= Tier-3 gate "
                f"({tier3}). Stake would buy fast-path exemption without any earned history. "
                f"Clamping ceiling to {tier3 - 5.0}."
            )
            self.stake_virtue_ceiling = tier3 - 5.0

        # Thresholds from config
        self.slash_threshold = rep_cfg.get("slash_threshold", 3)
        self.quorum_ratio = rep_cfg.get("quorum_ratio", 0.51)
        self.max_strikes = rep_cfg.get("max_strikes", 3)  # تغییر نام از slash_threshold
        # Per-reason override (2026-09-16, thesis-hub/docs/decisions.md same date,
        # idea-critique on lowering max_strikes to 1 project-wide): "poison_distance"
        # became reliable enough for 1-strike quarantine after today's two
        # engineering fixes (inherited-contamination exculpation, wide-layer RMS
        # guard), but "lazy_distance" has a DIFFERENT, essentially irreducible
        # false-positive floor - its low gate (`tau_low`) sits almost exactly on
        # the observed minimum of a 9,405-row honest calibration set
        # (thesis-hub/docs/consensus-signal-study.md, sec 6.2: 0.06% honest
        # leakage BY DESIGN, not a bug). Lowering `tau_low` itself was rejected
        # (no real headroom below the calibration data, and a 2026-09-08 attempt
        # to tighten this same adaptive band already caused a worse regression -
        # see `_current_gates`'s RATCHET GUARD). Keeping lazy_distance's own
        # quarantine bar higher restores the safety margin that `max_strikes:1`
        # removed globally, without touching the fragile gate itself. Falls back
        # to `self.max_strikes` for any reason not listed here.
        self.max_strikes_by_reason: Dict[str, int] = rep_cfg.get("max_strikes_by_reason", {})
        self.virtue_penalty = rep_cfg.get("virtue_penalty", 10.0)  # جریمه کلی
        self.min_virtue = rep_cfg.get("min_virtue", 0.0)
        self.reward_scale = rep_cfg.get("reward_scale", 0.01)

        # Sybil-resistant catch-up mechanism
        self.catch_up_enabled = rep_cfg.get("catch_up_enabled", True)
        self.catch_up_multiplier_low = rep_cfg.get("catch_up_multiplier_low", 1.5)   # For virtue < 30
        self.catch_up_multiplier_mid = rep_cfg.get("catch_up_multiplier_mid", 1.2)   # For 30 <= virtue < 60
        self.catch_up_max_budget_per_epoch = rep_cfg.get("catch_up_max_budget_per_epoch", 100.0)

        # Track catch-up rewards per epoch
        self._catch_up_spent_this_epoch = 0.0
        self._current_epoch = 0

        # M3 and z-score thresholds (must match consensus validation)
        self.z_score_threshold = abs(consensus_cfg.get("z_score_threshold", -3.0))
        self.isolation_threshold = consensus_cfg.get("isolation_threshold", 0.01)
        self.m3_threshold = consensus_cfg.get("m3_threshold", 0.01)  # نه 2.5!
        
        logger.info(
            f"[FARABI_REPUTATION] Initialized for node '{self.node_id}' with "
            f"quorum_ratio={self.quorum_ratio}, m3_threshold={self.m3_threshold}"
        )

    def handle_quarantine_proposal(self, proposal_tx: Dict[str, Any]) -> None:
        """Process incoming quarantine proposal from gossip"""
        proposal_id = proposal_tx.get("proposal_id")
        
        if not proposal_id:
            logger.warning("[QUARANTINE_PROPOSAL] Missing proposal_id")
            return
        
        # Ignore duplicates
        if proposal_id in self.quarantine_proposals:
            return
        
        # Store proposal
        self.quarantine_proposals[proposal_id] = {
            "target": proposal_tx["target_node"],
            "proposer": proposal_tx["proposer"],
            "evidence": proposal_tx["evidence"],
            "votes": {proposal_tx["proposer"]},  # Proposer auto-votes
            # `time` is the `time.time` function itself (see `from time
            # import time` above), so `time.time()` would AttributeError -
            # fixed to `time()`. Still wall-clock, not simulation time (this
            # method has no caller today - see thesis-hub/docs/decisions.md
            # 2026-09-03 - so it isn't a live source of run divergence, but
            # wire in real simulation time before ever calling this).
            "timestamp": proposal_tx.get("timestamp", time())
        }
        
        logger.warning(
            f"🗳️ [QUARANTINE_PROPOSAL] Received {proposal_id} "
            f"targeting {proposal_tx['target_node']} from {proposal_tx['proposer']}"
        )
        
        # Automatically evaluate and vote if evidence convinces me
        self._evaluate_and_vote(proposal_id)

    def _evaluate_and_vote(self, proposal_id: str) -> None:
        """Independent evaluation: if evidence is convincing, cast vote"""
        if proposal_id in self.my_votes:
            return  # Already voted
        
        if proposal_id not in self.quarantine_proposals:
            return
        
        prop = self.quarantine_proposals[proposal_id]
        evidence = prop["evidence"]
        
        # Apply same thresholds as in consensus validation
        m3 = evidence.get("m3", 0.0)
        z_score = evidence.get("z_score")
        reason = evidence.get("reason", "")
        
        should_vote = False
        
        # M3-based criterion
        if m3 > self.m3_threshold:
            should_vote = True
            logger.info(
                f"[VOTE_DECISION] M3={m3:.4f} > threshold={self.m3_threshold}, voting YES"
            )
        
        # Z-score criterion (for poison detection)
        if z_score is not None and abs(z_score) > self.z_score_threshold:
            should_vote = True
            logger.info(
                f"[VOTE_DECISION] |z_score|={abs(z_score):.4f} > threshold={self.z_score_threshold}, voting YES"
            )
        
        # Steal detection (M1-based)
        if "steal" in reason.lower():
            should_vote = True
            logger.info(f"[VOTE_DECISION] Steal detected in reason, voting YES")
        
        if should_vote:
            self.vote_quarantine(proposal_id, self.node_id)
            logger.warning(
                f"✅ [VOTE_CAST] {self.node_id} voted YES on {proposal_id} "
                f"(target={prop['target']}, reason={reason})"
            )
        else:
            logger.info(
                f"[VOTE_DECISION] Evidence insufficient for {proposal_id}, abstaining"
            )

    def vote_quarantine(self, proposal_id: str, voter_id: str) -> None:
        """Add a vote to an existing proposal"""
        if proposal_id not in self.quarantine_proposals:
            logger.warning(f"[VOTE] Proposal {proposal_id} not found")
            return
        
        self.quarantine_proposals[proposal_id]["votes"].add(voter_id)
        self.my_votes.add(proposal_id)

    def finalize_quarantine(self, total_validators: int) -> None:
        """Check all proposals and execute quarantine if quorum reached"""
        if not self.quarantine_proposals:
            return
        
        if total_validators is None:
         # Fallback: استفاده از تعداد گره‌های حاضر در ledger
            total_validators = max(len(self.ledger), 1)
            
        finalized = []
        required_votes = max(1, int(total_validators * self.quorum_ratio))
        
        for proposal_id, prop in self.quarantine_proposals.items():
            vote_count = len(prop["votes"])
            
            if vote_count >= required_votes:
                target = prop["target"]
                reason = prop["evidence"].get("reason", "consensus")
                
                # Execute actual quarantine
                self.slash_node(target, reason)
                
                logger.warning(
                    f"☣️ [QUARANTINE_FINALIZED] {target} quarantined with "
                    f"{vote_count}/{total_validators} votes (quorum={required_votes})"
                )
                
                finalized.append(proposal_id)
            else:
                # Log pending proposals
                logger.debug(
                    f"[QUARANTINE_PENDING] {proposal_id} has {vote_count}/{required_votes} votes"
                )
        
        # Clean up finalized proposals
        for pid in finalized:
            del self.quarantine_proposals[pid]

    def expire_old_proposals(self, max_age_seconds: float = 600.0) -> None:
        """Remove proposals older than max_age_seconds"""
        # Same `time` vs `time.time()` fix as handle_quarantine_proposal above.
        current_time = time()
        expired = []
        
        for proposal_id, prop in self.quarantine_proposals.items():
            age = current_time - prop["timestamp"]
            if age > max_age_seconds:
                expired.append(proposal_id)
                logger.info(
                    f"[QUARANTINE_EXPIRED] {proposal_id} expired after {age:.1f}s"
                )
        
        for pid in expired:
            del self.quarantine_proposals[pid]
            self.my_votes.discard(pid)

    def slash_node(self, node_id: str, reason: str) -> None:
        """Execute quarantine after consensus"""
        if node_id not in self.strikes:
            self.strikes[node_id] = 0

        self.strikes[node_id] += 1

        # Reduce virtue score
        if node_id in self.ledger:
            self.ledger[node_id] = max(self.min_virtue, self.ledger[node_id] - self.virtue_penalty)

        # Per-reason quarantine bar (see `max_strikes_by_reason` in __init__) -
        # falls back to the global `self.max_strikes` for any reason not
        # explicitly listed.
        threshold = self.max_strikes_by_reason.get(reason, self.max_strikes)

        # Quarantine if threshold reached
        if self.strikes[node_id] >= threshold:
            self.quarantined.add(node_id)
            logger.critical(
                f"☠️ [QUARANTINE] Node {node_id} quarantined after {self.strikes[node_id]} strikes "
                f"(reason: {reason}, threshold: {threshold})"
            )
        else:
            logger.warning(
                f"⚠️ [STRIKE] Node {node_id} received strike {self.strikes[node_id]}/{threshold} "
                f"(reason: {reason})"
            )

    def is_quarantined(self, node_id: str) -> bool:
        """Check if a node is quarantined"""
        return node_id in self.quarantined

    def reward_approval(self, node_id: str, reward: float = 1.0, current_time: float = 0.0) -> None:
        """
        Increase virtue score for honest behavior with Sybil-resistant catch-up.
        
        Args:
            node_id: ID of the node to reward.
            reward: Base reward amount.
            current_time: Current simulation time for epoch tracking.
        """
        farabi_cfg = self.config.get("protocols", {}).get("farabi", {})
        max_virtue = farabi_cfg.get("reputation", {}).get("max_virtue", 100.0)
        
        if node_id not in self.ledger:
            self.ledger[node_id] = self.initial_virtue_for(node_id)
        
        # Calculate catch-up multiplier
        multiplier = 1.0
        virtue = self.ledger[node_id]
        
        if self.catch_up_enabled and virtue < 60.0:
            # Check epoch boundary
            epoch = int(current_time / 100.0)  # 100s epochs
            if epoch != self._current_epoch:
                self._current_epoch = epoch
                self._catch_up_spent_this_epoch = 0.0  # Reset budget
            
            # Price the catch-up BEFORE deciding whether it fits the budget.
            #
            # This ordering was inverted from the commit that introduced the
            # mechanism (8b1b202, 2026-08-20) until 2026-09-06: the cost was
            # computed while `multiplier` was still 1.0, so it was always
            # exactly 0.0, the budget test `0 <= 100` was always true, and
            # `_catch_up_spent_this_epoch` never accumulated. The effect was that
            # catch-up was granted unconditionally to every node under virtue 60
            # and `catch_up_max_budget_per_epoch` was dead config. With
            # initial_virtue at 50 that means every node was boosted from the
            # start, and - the reason this matters beyond tidiness - a penalised
            # ATTACKER sits below 60 by construction, so it was collecting an
            # uncapped 1.2-1.5x tailwind back toward the Tier-3 gate.
            candidate = (
                self.catch_up_multiplier_low if virtue < 30.0
                else self.catch_up_multiplier_mid
            )
            # Cost is the EXTRA over the un-boosted reward, not the whole reward.
            catch_up_reward = reward * self.reward_scale * (candidate - 1.0)

            if self._catch_up_spent_this_epoch + catch_up_reward <= self.catch_up_max_budget_per_epoch:
                multiplier = candidate
                self._catch_up_spent_this_epoch += catch_up_reward
            # else: budget exhausted this epoch, multiplier stays 1.0
        
        # Apply reward with multiplier
        gained = reward * self.reward_scale * multiplier
        self.ledger[node_id] = min(self.ledger[node_id] + gained, max_virtue)
        
    def initial_virtue_for(self, node_id: str) -> float:
        """
        Entry reputation for a node that has no ledger history yet.

        In "constant" mode this is the flat `initial_virtue`. In "stake" mode the
        node's collateral is drawn deterministically from (run seed, node_id) — so
        the assignment is reproducible across runs and needs no registration hook —
        and mapped concavely to a virtue inside [floor, ceiling], with the ceiling
        held strictly below the Tier-3 fast-path gate (see __init__).
        """
        if self.initial_virtue_mode not in ("stake", "stake_sybil"):
            return self.initial_virtue

        if self.initial_virtue_mode == "stake_sybil" and node_id in self.attacker_ids:
            stake = min(max(self.sybil_stake, 1e-9), self.stake_max)
        elif node_id in self.initial_stake:
            stake = self.initial_stake[node_id]
        else:
            digest = hashlib.sha256(f"{self._stake_seed}:{node_id}".encode()).digest()
            u = int.from_bytes(digest[:8], "big") / float(1 << 64)  # uniform [0,1)
            lo, hi = max(self.stake_min, 1e-9), max(self.stake_max, self.stake_min + 1e-9)
            stake = lo * ((hi / lo) ** u)  # log-uniform: many small, few large holders
            self.initial_stake[node_id] = stake

        # Concave: quadrupling stake only doubles the trust it buys.
        frac = math.sqrt(stake / max(self.stake_max, 1e-9))
        virtue = self.stake_virtue_floor + frac * (self.stake_virtue_ceiling - self.stake_virtue_floor)
        return min(max(virtue, self.min_virtue), self.stake_virtue_ceiling)

    def set_attacker_ids(self, attacker_ids: Set[str]) -> int:
        """Hand over the attacker identities for "stake_sybil" and re-seed every
        ledger entry created before this call (entries still at their entry value
        are replaced; returns how many entries had already moved, which should be 0
        when called at the first topology update)."""
        if self.initial_virtue_mode != "stake_sybil" or self.attacker_ids_set:
            return 0
        moved = 0
        old_entry = {n: self.initial_virtue_for(n) for n in self.ledger}
        self.attacker_ids = set(attacker_ids)
        self.attacker_ids_set = True
        for n in list(self.ledger):
            if abs(self.ledger[n] - old_entry[n]) > 1e-9:
                moved += 1
                continue
            self.ledger[n] = self.initial_virtue_for(n)
        return moved

    def get_virtue_score(self, node_id: str) -> float:
        """Retrieve current virtue score"""
        if node_id not in self.ledger:
            self.ledger[node_id] = self.initial_virtue_for(node_id)
        return self.ledger.get(node_id, 0.0)

    def get_average_network_virtue(self) -> float:
        """Returns the average virtue score across all known nodes."""
        if not self.ledger:
            return self.initial_virtue
        return sum(self.ledger.values()) / len(self.ledger)

    def get_total_network_stake(self) -> float:
        """Returns the sum of all virtue scores in the ledger."""
        return max(sum(self.ledger.values()), 1.0)
    
    def get_reputation_score(self, node_id: str) -> float:
        """
        Returns a normalized reputation score between 0.0 and 1.0.
        Used for T²RC tier routing (compared against tier3_threshold=0.8).
        Normalization: virtue / 100.0 (initial virtue is 50, rewards increase it).
        """
        virtue = self.get_virtue_score(node_id)
        return min(max(virtue / 100.0, 0.0), 1.0)
    
    def apply_consensus_slash(self, node_id: str, reason: str) -> None:
        """
        On-tangle consensus verdict: immediate quarantine in the LOCAL ledger.
        Consensus IS the verdict — no strike accumulation needed.
        """
        self.quarantined.add(node_id)
        self.strikes[node_id] = self.max_strikes
        self.ledger[node_id] = max(
            self.min_virtue,
            self.ledger.get(node_id, self.initial_virtue_for(node_id)) - self.virtue_penalty
        )
        logger.critical(
            f"☠️ [CONSENSUS SLASH] Node {node_id} quarantined by on-tangle "
            f"consensus in local ledger (reason: {reason})"
        )

    def apply_approver_penalty(self, node_id: str, reason: str) -> None:
        """
        Smaller penalty for a node that approved (referenced as a DAG
        parent) a block later confirmed malicious by consensus slash.
        Deliberately lighter than apply_consensus_slash - approving isn't
        the malicious act itself, but until now it carried zero cost,
        which made "voting" for a parent free regardless of how carefully
        it was vetted. This gives approvers a real incentive to be
        careful about what they choose to build on.
        """
        penalty = self.virtue_penalty * 0.25
        self.ledger[node_id] = max(
            self.min_virtue,
            self.ledger.get(node_id, self.initial_virtue_for(node_id)) - penalty
        )
        logger.warning(
            f"⚠️ [APPROVER_PENALTY] Node {node_id} penalized {penalty:.2f} virtue "
            f"for approving a since-slashed block (reason: {reason})"
        )