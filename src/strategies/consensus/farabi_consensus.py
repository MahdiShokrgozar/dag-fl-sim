"""
Module: farabi_consensus.py
Description: 
    Implements the FARABI Consensus using T²RC (Tiered-Trust Reactive Consensus) and 
    a Parameter-Free Phase-Aware Unified Farabi Score (UFS).
    Incorporates Spot-Checking, Sigmoid-based Novelty evaluation, and Graduated Slashing.
"""

import hashlib
import logging
import math
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from src.strategies.consensus.base import ConsensusStrategy
from src.utils.ipfs_utils import download_model_from_ipfs, MockIPFS
from src.metrics.signal_probe import SignalProbe
from src.metrics.trust_signals import TrustSignalExtractor
from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)
SLASH_EVIDENCE_TYPE = "slash_evidence"
QUARANTINE_PROPOSAL_TYPE = "quarantine_proposal"

# Thin-base skip sanity ceiling (2026-09-10, thesis-hub/docs/decisions.md same date):
# 4ee9a37's own investigation recorded the two clusters this skip is meant to tell
# apart - honest thin-base noise topped out at 1.934, genuine poisoning started at
# ~700. A distance this far past tau_high cannot plausibly be thin-base noise at
# any base size, so the skip must not apply to it even when base_size is thin.
THIN_BASE_DISTANCE_SANITY_CEILING = 10.0

import json
import hashlib

def generate_metadata_commitment(class_accuracies: dict, accuracy: float) -> str:
    """
    Generate a hardware-invariant commitment hash for metadata.
    
    Quantizes floats to 3 decimal places to eliminate CPU/GPU floating-point
    discrepancies, and sorts keys to ensure deterministic JSON serialization.
    
    Args:
        class_accuracies: Dictionary of per-class accuracies.
        accuracy: Overall accuracy.
        
    Returns:
        SHA256 hex digest of quantized metadata.
    """
    # Quantize to 3 decimal places (sufficient for accuracy percentages)
    quantized_data = {
        "acc": round(accuracy, 3),
        "classes": {str(k): round(v, 3) for k, v in sorted(class_accuracies.items())}
    }
    
    # Deterministic JSON serialization
    json_str = json.dumps(quantized_data, sort_keys=True)
    
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

class FarabiConsensus(ConsensusStrategy):
    """
    Evaluates peer models in a decentralized Federated Learning DAG.
    Implements T²RC to optimize IPFS bandwidth for trusted nodes, while maintaining
    strict zero-trust validation for suspicious or new peers.
    """

    #: How often (in accepted distances) to log the current gate band.
    GATE_BAND_LOG_EVERY = 25

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        farabi_cfg = self.config.get("protocols", {}).get("farabi", {})
        rep_cfg = farabi_cfg.get("reputation", {})
        consensus_cfg = farabi_cfg.get("consensus", {})
        
        # Trust Score filter parameters (additional, not replacement)
        self.use_trust_score_filter = consensus_cfg.get("use_trust_score_filter", False)
        self.trust_threshold = consensus_cfg.get("trust_threshold", 0.5)
            
        # T²RC (Tiered-Trust) Configuration
        self.tier3_threshold = consensus_cfg.get("tier3_threshold", 0.8)  # High Trust
        self.spot_check_prob = consensus_cfg.get("spot_check_prob", 0.05) # 5% chance to deep-inspect Tier 3
        self.proposer_reward_ratio = consensus_cfg.get("proposer_reward_ratio", 0.05)
        self.catastrophic_drop_threshold = consensus_cfg.get("catastrophic_drop_threshold", -0.30)
        self.z_score_threshold = consensus_cfg.get("z_score_threshold", -3.0)
        
        # M1 (Novelty) Sigmoid Configuration (Tuned for typical distances ~ 0.008)
        self.novelty_steepness = consensus_cfg.get("novelty_steepness", 10.0)
        self.novelty_midpoint = consensus_cfg.get("novelty_midpoint", 0.15)
        # ✅ موقت: حاشیه کالیبره از لاگ‌ها (صادق‌ها M2≥0.83، مهاجم M2≤0.6)
        # TODO: جایگزینی با نسخه تطبیقی median/MAD
        self.divergence_m2_threshold = consensus_cfg.get("divergence_m2_threshold", 0.7)
        
        # --- Inherited-contamination exculpation (2026-09-16, one-hop version -
        # thesis-hub/docs/decisions.md and research-direction.md item 14) -------
        # A "poison_distance" accusation is exculpated (block still rejected from
        # aggregation, but the proposer is NOT accused) when its raw_distance is
        # already explained by its own direct aggregation_refs - i.e. it
        # aggregated an already-anomalous, not-yet-quarantined ancestor and added
        # nothing new itself. Deliberately single-hop (compares only against
        # direct parents' own recorded raw_distance, not a recursive walk to the
        # true origin) - see decisions.md for why this scope was chosen first.
        # UNCALIBRATED placeholder - needs the same 3-seed sweep IronForge's
        # slashing_threshold got before any number here is cited in the paper.
        self.contamination_tolerance = float(consensus_cfg.get("contamination_tolerance", 5.0))

        # General Thresholds
        self.isolation_threshold = consensus_cfg.get("isolation_threshold", 0.01)
        self.frankenstein_tolerance = consensus_cfg.get("frankenstein_tolerance", 1.0)
        self.base_bw_mbps = consensus_cfg.get("base_bw_mbps", 50.0)
        self.network_overhead_sec = consensus_cfg.get("network_overhead_sec", 0.2)
        self.eval_compute_delay = consensus_cfg.get("eval_compute_delay", 0.05)
        self.neutral_vote_weight = consensus_cfg.get("neutral_vote_weight", 0.05)
        self.epsilon = consensus_cfg.get("epsilon", 1e-9)
        
        # Weights for Security Score Calculation
        self.w_exp_novelty = consensus_cfg.get("w_exp_novelty", 1.5)
        self.w_exp_utility = consensus_cfg.get("w_exp_utility", 0.5)
        self.w_plt_novelty = consensus_cfg.get("w_plt_novelty", 0.5)
        self.w_plt_utility = consensus_cfg.get("w_plt_utility", 1.5)
        
        # Smart Steepness Boundaries (Zero-Trust Phase Detection)
        self.strict_steepness = consensus_cfg.get("strict_steepness", 10.0)    
        self.lenient_steepness = consensus_cfg.get("lenient_steepness", 2.0)   
        self.phase_detection_threshold = consensus_cfg.get("phase_detection_threshold", 0.5)
        self.MIN_PERSISTENCE = rep_cfg.get("max_strikes", 3)  # ✅ از max_strikes
        
        self._flag_counter: Dict[str, int] = {}
        
        # Robust Statistics Sliding Window (Stores M2)
        self.history_window_size = 20
        self.m2_score_history: List[float] = []

        # --- S1 two-gate anomaly detection (2026-09-05) -------------------------
        # Replaces the m1/m2 sigmoid + z-score rules below. Calibrated on 9,405
        # honest-only probe rows from plan 21: the relative L2 distance of an honest
        # update lives in a tight band, a lazy free-rider republishes its parent so
        # the distance is exactly 0, and a poisoner blows far past the top of the
        # band. Measured on held-out attack cells: lazy 100% / poisoning 100% caught,
        # with 0.06% / 0.00% honest leakage. See thesis-hub/docs/consensus-signal-study.md.
        #
        # Backdoor sits entirely INSIDE the honest band ([0.003, 0.172] vs
        # [0.0027, 0.227]) and is provably not separable by any threshold on this
        # signal — that is a documented limitation, not a tuning gap.
        gate_cfg = consensus_cfg.get("anomaly_gates", {})
        self.anomaly_mode = consensus_cfg.get("anomaly_mode", "gates")  # "gates" | "legacy"
        self.gate_tau_low = float(gate_cfg.get("tau_low", 0.00274))
        self.gate_tau_high = float(gate_cfg.get("tau_high", 0.2613))
        self.gate_adaptive = bool(gate_cfg.get("adaptive", True))
        self.gate_history_size = int(gate_cfg.get("history_size", 200))
        # Lowered 30 -> 10 on 2026-09-09. The 2026-09-09 cold-start-window fix
        # (see _in_gate_cold_start) made this constant costlier than before:
        # once nothing gets admitted into the window until min_history gate
        # evaluations pass, ALL protection (not just the high gate's own
        # skip) is void for that whole stretch. Measured on a real 12-node,
        # 400s mixed_attack run: min_history=30 translated to ~50-55 real
        # simulated seconds (~13% of the run) with zero poison-distance
        # enforcement per validator, active poisoners present from t=0 -
        # the dominant suspect for a ~50-point accuracy gap versus the
        # (now-unverifiable) pre-regression baseline. 30 was never
        # calibrated for a specific "settling in" reason - it was inherited
        # from when the window populated from block 1 with no lockout at
        # all, a context this constant no longer lives in. Not swept
        # (5/10/15/30) due to Kaggle cost; picked one lower value and
        # re-verified the node_6/node_9 regression guard still holds at 10
        # rather than guessing blind. See thesis-hub/docs/decisions.md,
        # 2026-09-09.
        #
        # Known open risk, NOT addressed here: this is a block-COUNT, and
        # its translation to real time depends on block-production rate,
        # which is itself network-size-dependent - untested whether this
        # value's real cost is scale-invariant. Tracked alongside
        # thesis-hub/docs/research-direction.md item 9 (same failure class:
        # a constant whose real cost was only ever measured at one scale).
        self.gate_min_history = int(gate_cfg.get("min_history", 10))
        # W1 (2026-09-24, thesis-hub/docs/decisions.md same date): cold start
        # used to switch the high gate off for ANY distance. With correct L2
        # timing (blocks reach validators later), half the validators were
        # still in cold start when evidence arrived and abstained on a
        # distance of ~725 against tau_high ~0.26, so the quorum never formed.
        # When on, cold start only suppresses distances at or below the same
        # sanity ceiling the thin-base skip already uses. Default off keeps
        # every earlier run reproducible.
        self.gate_cold_start_sanity_ceiling = bool(gate_cfg.get("cold_start_sanity_ceiling", False))
        # W4 (2026-09-26, thesis-hub/docs/decisions.md same date): the lazy gate
        # only fires on an EXACT zero distance, but its reference is the
        # median of the parents while Farabi-Prox publishes a virtue x prox x
        # staleness weighted MEAN of them. With one parent the two coincide
        # (6/6 single-parent lazy blocks caught); with several they never do
        # (73/73 multi-parent lazy blocks at 0.033-0.05, never caught). When
        # on, the proposer declares the coefficients it actually used
        # (propose_block) and the lazy gate compares against that exact
        # combination - k floats per block, no extra fetch. The poison gate
        # keeps the robust median reference.
        self.declared_aggregation_base = bool(gate_cfg.get("declared_aggregation_base", False))
        # Multiplicative margins around the observed honest extremes.
        self.gate_low_margin = float(gate_cfg.get("low_margin", 0.5))
        self.gate_high_margin = float(gate_cfg.get("high_margin", 1.15))
        # The adaptive band may widen past the calibrated one but never narrow
        # inside it. See _current_gates for why; set False to reproduce the
        # plan-22 (unclamped) behaviour for an A/B.
        self.gate_clamp_to_bootstrap = bool(gate_cfg.get("clamp_to_bootstrap", True))
        # Path B (thesis-hub/docs/decisions.md, 2026-09-17): anchor the
        # calibration window to each block's DAG-topological depth instead of
        # this validator's local arrival order. Two validators that have
        # accepted the same set of blocks now derive the SAME window
        # regardless of the order network delivery happened to hand those
        # blocks to each -- Path A (deployed first) only disclosed the
        # arrival-order divergence as a limitation; this removes it. Set to
        # False to reproduce the old arrival-order behaviour for an A/B.
        self.gate_dag_anchored_window = bool(gate_cfg.get("dag_anchored_window", True))
        # Distances of ACCEPTED blocks only. Feeding rejected blocks back in would
        # let a poisoner drag the band toward itself.
        # Each entry is (depth, block_hash, distance) when gate_dag_anchored_window
        # is True (eviction below drops the lowest (depth, hash), i.e. the
        # topologically oldest entry, not the least-recently-appended one); a
        # bare distance list otherwise, preserving the pre-Path-B behaviour.
        self._distance_history: List = []
        # Per-validator memo of each block's own DAG-topological depth (0 for
        # genesis, else 1 + max(parent depths)), recomputed independently from
        # this validator's own local DAG view rather than trusted from a
        # self-reported field on the block -- same evidence-verification
        # principle as recomputing `d` in Sec. III-D. Unbounded for the life
        # of one simulation run, same lifetime as the other per-run caches in
        # this class (e.g. `_proposer_last_cid`).
        self._depth_cache: Dict[str, Optional[int]] = {}
        # Total non-genesis blocks this validator has gate-evaluated, accepted or
        # not. Deliberately independent of `_distance_history`: cold-start length
        # used to be measured by history length, which stalls forever if every
        # block during cold-start gets rejected (chicken-and-egg). See
        # `_in_gate_cold_start` and thesis-hub/docs/decisions.md, 2026-09-09.
        self._gate_blocks_seen = 0
        
        self.plagiarism_window = consensus_cfg.get("plagiarism_window", 100)
        self._fp_registry: Dict[str, str] = {}   # strong_hash -> author_id
        # Default False since 2026-09-12 -- TEMPORARY, must be repaired for the
        # main papers/thesis, not left off. Full rationale and the agreed
        # four-condition redesign are at the gate itself in
        # `_validate_deep_inspection` and in thesis-hub/docs/decisions.md.
        self.enable_steal_detection = consensus_cfg.get("enable_steal_detection", False)
        # Model-stealing detection, rebuilt (thesis-hub/docs/decisions.md
        # 2026-09-26; research-direction.md item 13, the four-condition design
        # agreed 2026-09-12). Replaces the "first publisher owns the hash"
        # registry above, which was measured backwards. A CID is the SHA-256 of
        # the serialized weights, so a verbatim copy carries the victim's CID
        # in its header: candidates are found from headers alone, on the fast
        # Tier-3 path too, with no model download. Theft is proven only when
        # every peer can re-derive, from content-addressed headers:
        #   (1) accused CID == victim CID
        #   (2) victim became ready strictly before the accused
        #   (3) different authors
        #   (4) the accused is not republishing its own aggregation base
        #       (that is the lazy signature, handled by the lazy gate) - read
        #       from its declared aggregation coefficients (W4). A block with
        #       no aggregation_refs never fires (cold-start carve-out).
        # The index is only a lookup for candidates; it is never part of a
        # verdict, and the older/newer order is by ready time, not arrival.
        self.steal_detection_cid = bool(consensus_cfg.get("steal_detection_cid", False))
        self._cid_index: Dict[str, str] = {}   # cid -> hash of the oldest block seen with it

        self.storage_config = self.config.get("storage", {"ipfs_mock_enabled": True})
        self.node = None  
        self.local_hot_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        self._probe_enabled: bool = bool(consensus_cfg.get("signal_probe", False))
        self.signal_extractor: TrustSignalExtractor = TrustSignalExtractor()
        self.signal_probe: Optional[SignalProbe] = None
        self._proposer_last_cid: Dict[str, str] = {}

        # --- Signal-symmetric evidence verification (2026-09-08) -----------------
        # Every entry pairs the SAME (materials, compute, gate) triple used to
        # DETECT an anomaly with what a peer uses to INDEPENDENTLY VERIFY the
        # resulting evidence. Before this registry, `_validate_slash_evidence`
        # pattern-matched on the `reason` string and trusted self-reported
        # numbers (m1/m2/acc) from the evidence publisher - a node could forge
        # those numbers, and "lazy_distance"/"poison_distance" (the S1 gates'
        # own output) matched no branch at all, so gate-detected laziness could
        # publish evidence forever without ever reaching quorum. See
        # thesis-hub/docs/decisions.md, 2026-09-08, for the full derivation and
        # why this is scoped to "recompute existing signals", not "detect any
        # unanticipated attack" (no evidence scheme can verify a signal nobody
        # measures).
        self._signal_registry: Dict[str, Dict[str, Any]] = self._build_signal_registry()

        # --- Virtue-free detection base (2026-09-08, thesis-hub/docs/decisions.md "option A") ---
        # `_aggregate_parents_if_needed`'s base is virtue-weighted and, in `dual`
        # selection mode, re-picked from whatever is CURRENTLY finalized - both
        # are time-varying, so re-deriving it later (independent verification,
        # possibly long after original detection) does not reproduce the
        # original value and can flag genuinely honest nodes. Kept for the
        # (currently dead - `farabi_processor.py` hardcodes `is_full_model=True`
        # - but not deleted) delta-reconstruction fallback. The S1 distance
        # signal itself now compares against `_detection_base_from_parents`
        # instead: coordinate-wise median/trimmed-mean over the SAME
        # `aggregation_refs`, with no Virtue lookup and no re-selection of
        # "currently finalized" anything, so it is reproducible bit-for-bit by
        # any peer at any time. Reuses FarabiRobustAggregator's fetch pipeline
        # only for its virtue-free stages (`_lazy_fetch_weights`,
        # `_aggregate_feature_layers`) - deliberately skips
        # `_resolve_aggregation_inputs` (re-picks fresh anchors under `dual`,
        # reintroducing the exact instability being removed here) and
        # `_filter_weight_space_anomalies` (anchors on the CALLING validator's
        # own current `node.weights`, an equally time/observer-varying signal).
        self._detection_base_helper = FarabiRobustAggregator(config)

    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """
        Wraps the payload with FARABI consensus metadata including commitment hash.
        
        Args:
            payload: Block payload containing model weights and metadata.
            node_id: ID of the proposing node.
            
        Returns:
            Block dictionary with metadata commitment.
        """
        # Generate commitment hash for metadata
        metadata_commitment = generate_metadata_commitment(
            class_accuracies=payload.get("class_accuracies", {}),
            accuracy=payload.get("accuracy", 0.0)
        )
        
        if self.declared_aggregation_base and self.node is not None:
            coefficients = getattr(self.node, "last_aggregation_coefficients", None)
            if coefficients:
                payload["aggregation_coefficients"] = [[h, float(c)] for h, c in coefficients]

        return {
            "payload": payload,
            "proposer": node_id,
            "type": "farabi_v2_t2rc",
            "metadata_commitment": metadata_commitment  # NEW: Commitment hash
        }

    def validate_block(self, block: Dict[str, Any]) -> bool:
        """
        Validates a received block based on T²RC consensus rules.
        Routes to O(1) checks for trusted nodes or Deep Inspection for others.
        """
        block_type = block.get("type", "")
        
        if "timestamp" not in block:
            block["timestamp"] = getattr(self.node.simulator, "current_time", 0.0) if self.node and hasattr(self.node, "simulator") else 0.0
    
        if block_type == SLASH_EVIDENCE_TYPE:
            return self._process_slash_evidence(block)
            
        if "payload" not in block or "proposer" not in block:
            logger.error(f"🚨 [VALIDATION] Block missing required fields: payload or proposer")
            return False
            
        if not hasattr(self, 'node') or self.node is None:
            logger.error(f"🚨 [VALIDATION] Node not initialized")
            return True 
        
       
        
        proposer_id = block["proposer"]
        payload = block["payload"]
        
        if hasattr(self.node, 'l2_strategy') and hasattr(self.node.l2_strategy, 'locally_slashed_nodes'):
            if proposer_id in self.node.l2_strategy.locally_slashed_nodes:
                logger.warning(
                    f"⛔ [SLASH-GATE] Block from slashed node {proposer_id} rejected in validation."
                )
                return False
        
        # ✅ Quarantine Gate
        if hasattr(self.node, 'reputation') and self.node.reputation.is_quarantined(proposer_id):
            logger.warning(
                f"⛔ [QUARANTINE-GATE] Block from quarantined node {proposer_id} rejected in validation."
            )
            return False
        
        # Self-proposed blocks are implicitly trusted locally
        if proposer_id == self.node.node_id:
            logger.info(f"✅ [VALIDATION] Self-proposed block from {proposer_id} is implicitly trusted.")
            return True

        # Fetch Node Reputation for T²RC routing
        if self.steal_detection_cid and self._check_cid_theft(block, proposer_id):
            return False

        reputation_module = getattr(self.node, 'reputation', None)
        proposer_virtue_raw = reputation_module.get_virtue_score(proposer_id) if reputation_module else 50.0
        proposer_virtue_norm = reputation_module.get_reputation_score(proposer_id) if reputation_module else 0.5        
        is_spot_check = random.random() < self.spot_check_prob
        
        # T²RC Gatekeeper: Route based on trust tier
        if proposer_virtue_norm >= self.tier3_threshold and not is_spot_check:
            logger.info(f"⚡ [T2RC] Fast-tracking block in tier 3 from trusted node {proposer_id} (Virtue: {proposer_virtue_norm:.2f})")
            return self._validate_trusted_tier3(block, payload, proposer_id, proposer_virtue_norm, reputation_module)
        else:
            if is_spot_check:
                logger.warning(f"🔍 [T2RC] Spot Check triggered for highly trusted node {proposer_id}.")
            return self._validate_deep_inspection(block, payload, proposer_id, proposer_virtue_norm, reputation_module)

    def _validate_trusted_tier3(self, block: Dict[str, Any], payload: Dict[str, Any], proposer_id: str, virtue: float, reputation_module: Any) -> bool:
        """
        O(1) Validation path for highly trusted nodes. Bypasses IPFS download.
        """
        claimed_classes = payload.get("class_accuracies", {})
        eval_classes = getattr(self.node, 'per_class_accuracy', {})
        
        # O(1) Check 1: M3 Phase Alignment
        if not claimed_classes or not eval_classes:
            # Missing metadata is not evidence of dishonesty; use a neutral score
            # instead of penalizing to zero, and log which side was empty for debugging.
            m3_alignment = 0.5
            logger.warning(
                f"⚡ [T2RC] Node {proposer_id} | M3 metadata missing "
                f"(claimed_empty={not claimed_classes}, eval_empty={not eval_classes}) "
                f"-> using neutral alignment 0.5"
            )
        else:
            m3_alignment = self._calculate_class_overlap(claimed_classes, eval_classes)
            
        logger.warning(f"⚡ [T2RC] Node {proposer_id} | M3 Alignment: {m3_alignment:.4f} | Virtueـnorm: {virtue:.2f}")
        # O(1) Check 2: Plausible claimed accuracy (Heuristic)
        claimed_acc = payload.get("accuracy", 0.0)
        parent_hashes = block.get('parents', payload.get('parents', []))
        
        if claimed_acc < 0.1: # Absurdly low accuracy for a trusted node
            block["_validation_receipt"] = {
                "status": "rejected",
                "reason": "suspicious_metadata"
            }
            
            self._publish_slash_evidence(
                target_id=proposer_id,
                reason="suspicious_metadata",
                evidence={"claimed_acc": claimed_acc, "validator": self.node.node_id},
                signal_id="suspicious_metadata",
                block_ref={"claimed_acc": claimed_acc, "proposer": proposer_id},
            )
            return False

        # Apply approval weights directly without deep math
        final_approval_weight = 10.0 * virtue * m3_alignment 
        self._assign_approval_weights(payload, parent_hashes, final_approval_weight, proposer_id, reputation_module)
        
        logger.info(f"⚡ [T2RC] Fast-tracked block from _validate_trusted_tier3 "
                    f"{proposer_id} Virtue: {virtue:.2f}), "
                    f"m3 alignment: {m3_alignment:.2f}, "
                    f"final approval weight: {final_approval_weight:.2f} validated by node {self.node.node_id} successfully.")
        return True

    def _validate_deep_inspection(self, block: Dict[str, Any], payload: Dict[str, Any], proposer_id: str, virtue: float, reputation_module: Any) -> bool:
        """
        Standard/Strict Validation path involving IPFS downloads and full cryptographic math.
        """
        self._bind_probe()

        ipfs_cid = payload.get("ipfs_cid")
        claimed_fingerprint = payload.get("model_fingerprint", 0.0)
        is_full_model = payload.get("is_full_model", False)
        parent_hashes = block.get('parents', payload.get('parents', []))
        is_genesis_block = len(parent_hashes) == 0

        # The base must be rebuilt from the set the AUTHOR aggregated, which is
        # not necessarily the parent set. In dual selection the parents are tips
        # (consensus duty) while the aggregation base is drawn from finalized
        # blocks, and rebuilding from the parents there would make every honest
        # block fail its own consistency check. Falls back to `parents` for
        # blocks written before this field existed.
        aggregation_refs = payload.get('aggregation_refs') or block.get('aggregation_refs') or parent_hashes

        if not ipfs_cid:
            return False

        # 1. Lazy IPFS Download (Only happens here)
        downloaded_weights = self._fetch_data_from_ipfs(ipfs_cid)
        if not downloaded_weights:
            logger.warning(f"🚨 [VALIDATION] Failed to fetch model from IPFS for {proposer_id}. CID: {ipfs_cid}")
            return False

        base_w = self._aggregate_parents_if_needed(aggregation_refs, is_genesis_block, proposer_id)
        if base_w is None and not is_genesis_block:
            logger.warning(f"🚨 [VALIDATION] Soft reject {proposer_id}: Missing parents.")
            return False # Soft reject for missing parents
            
        # 2. Reconstruct Model
        reconstructed_weights = downloaded_weights if is_full_model else self._reconstruct_full_model(base_w, downloaded_weights)
        if not reconstructed_weights:
            logger.warning(f"🚨 [VALIDATION] Failed to reconstruct model for {proposer_id}.")
            return False
            
        self.local_hot_cache[f"{ipfs_cid}_full"] = reconstructed_weights

        # --- Hash-collision plagiarism gate: DISABLED BY DEFAULT (2026-09-12) ---------
        # TEMPORARY, and it must not stay this way. Turned off to unblock the
        # conference submission; the mechanism is to be repaired for the main
        # papers and the thesis. See thesis-hub/docs/decisions.md, 2026-09-12,
        # which carries the agreed four-condition redesign and the student's
        # explicit instruction that this be fixed afterwards, not dropped.
        #
        # Why it is off rather than trusted:
        #   - Measured precision is 0%. Plan 23 group 3 contains NO stealing
        #     nodes (honest 0.6 / lazy 0.2 / poisoning 0.2), so all 34 (unified)
        #     and 55 (dual) PLAGIARISM events in seed_43 are false by construction.
        #   - It is not merely noisy, it is backwards. `_fp_registry` awards
        #     authorship of a weight-state to whoever publishes it FIRST, and
        #     attackers that skip training publish 1.40x more blocks (lazy: 141
        #     vs honest: ~101) and start sooner (1.33s vs 2.07s). So when an
        #     honest node is convicted, the registered "original author" is a
        #     malicious node 68-85% of the time against a 33.3% base rate,
        #     consistently across both modes and both sides of the 2026-09-12
        #     quorum fix. The less work a node does, the more likely it is to be
        #     recorded as the victim -- and an attacker can weaponise that
        #     deliberately to frame honest peers.
        #   - Verification is not reproducible: `_gate_steal_hash_collision`
        #     consults the VERIFIER's own local registry, so two honest peers
        #     can reach opposite verdicts from gossip arrival order alone. That
        #     violates the evidence-verification contract of 2026-09-08.
        #
        # Leaving it enabled costs real accuracy now that containment actually
        # works (2026-09-12 quorum fix): false positives are enforced, not just
        # recorded. Skipping the hash also removes a per-validation cost.
        if self.enable_steal_detection:
            strong_hash = self._strong_model_hash(reconstructed_weights)
            victim = self._fp_registry.get(strong_hash)

            if victim is not None and victim != proposer_id:
                logger.critical(
                    f"🕵️ [PLAGIARISM] {proposer_id} published an EXACT copy of {victim}'s model!"
                )
                self._publish_slash_evidence(
                    target_id=proposer_id,
                    reason="steal",
                    evidence={
                        "strong_hash": strong_hash,
                        "victim": victim,
                        "m1": 0.0,
                        "reason": "steal",
                        "validator": self.node.node_id,
                    },
                    signal_id="steal_hash_collision",
                    block_ref={
                        "ipfs_cid": ipfs_cid,
                        "aggregation_refs": aggregation_refs,
                        "aggregation_coefficients": payload.get("aggregation_coefficients"),
                        "is_genesis": is_genesis_block,
                        "is_full_model": is_full_model,
                        "proposer": proposer_id,
                    },
                )
                block["_validation_receipt"] = {"status": "rejected", "reason": "steal"}
                return False

            # ثبت اثرانگشت این بلاک (با پنجره محدود)
            self._fp_registry[strong_hash] = proposer_id
            if len(self._fp_registry) > self.plagiarism_window:
                self._fp_registry.pop(next(iter(self._fp_registry)))
    
        # 3. Frankenstein Trap (Basic checksum verification)
        if not self._verify_frankenstein_trap(reconstructed_weights, claimed_fingerprint):
            logger.warning(f"⚠️ [VALIDATION] Frankenstein Trap triggered for {proposer_id}.")

            self._publish_slash_evidence(
                target_id=proposer_id,
                reason="frankenstein_plagiarism",
                evidence={
                    "claimed_fingerprint": claimed_fingerprint,
                    "validator": self.node.node_id
                },
                signal_id="frankenstein_checksum",
                block_ref={
                    "ipfs_cid": ipfs_cid,
                    "aggregation_refs": aggregation_refs,
                    "is_genesis": is_genesis_block,
                    "is_full_model": is_full_model,
                    "proposer": proposer_id,
                    "claimed_fingerprint": claimed_fingerprint,
                },
            )
            return False
            
        if hasattr(self.node, 'telemetry'):
            # Scaled by the node's effective capability, like every other compute
            # charge in the simulator (FarabiSelection, IronForgeConsensus,
            # IronForgeTipSelector). This was a flat constant, which both
            # contradicted explicit assumption 1 of the thesis (heterogeneous
            # devices) and made the cost comparison against IronForge uneven:
            # IronForge's PoL charge divides by capacity, so a slow node paid
            # more there but a fixed amount here.
            # See thesis-hub/docs/decisions.md, 2026-09-12.
            cap = getattr(self.node, 'capacity', 1.0)
            delay_multiplier = getattr(self.node, 'delay_multiplier', 1.0)
            effective_capacity = cap / max(0.01, delay_multiplier)
            self.node.telemetry.register_compute(
                self.eval_compute_delay / max(0.1, effective_capacity)
            )

        # 3.5. Verify Metadata Commitment (Prevent Metadata Tampering)
        claimed_commitment = block.get("metadata_commitment", "")
        if claimed_commitment:
            actual_commitment = generate_metadata_commitment(
                class_accuracies=payload.get("class_accuracies", {}),
                accuracy=payload.get("accuracy", 0.0)
            )
            
            if claimed_commitment != actual_commitment:
                logger.critical(
                    f"🚨 [METADATA TAMPERING] {proposer_id} claimed different metadata!"
                )
                self._publish_slash_evidence(
                    target_id=proposer_id,
                    reason="metadata_tampering",
                    evidence={
                        "claimed_commitment": claimed_commitment,
                        "actual_commitment": actual_commitment,
                        "validator": self.node.node_id
                    },
                    signal_id="metadata_commitment_mismatch",
                    block_ref={
                        "claimed_commitment": claimed_commitment,
                        "class_accuracies": payload.get("class_accuracies", {}),
                        "accuracy": payload.get("accuracy", 0.0),
                        "proposer": proposer_id,
                    },
                )
                return False
            
        # 4. Deep Math Evaluation (UFS)
        # actual_tested_acc = block.get("ground_truth", {}).get("accuracy", payload.get("accuracy", 0.0))
        # parents_tested_acc = block.get("ground_truth", {}).get("parent_accuracy", 0.0)

        oracle = getattr(self.node.simulator, 'oracle_state', {})
        block_hash = block.get("hash", block.get("id", ""))
        ground_truth = oracle.get(block_hash, {})
        actual_tested_acc = ground_truth.get("accuracy", payload.get("accuracy", 0.0))
        parents_tested_acc = ground_truth.get("parent_accuracy", 0.0)

        claimed_classes = payload.get("class_accuracies", {i: 1.0 for i in range(10)})
        eval_classes = getattr(self.node, 'per_class_accuracy', {i: 1.0 for i in range(10)})

        # Distance is measured against the virtue-free detection base, not
        # `base_w` (virtue-weighted, only kept for the dormant delta-reconstruction
        # path above) - see `_detection_base_from_parents` docstring and
        # thesis-hub/docs/decisions.md, 2026-09-08. `None` (parent fetch failed)
        # is treated the same as "no base" (falsy {}), which `evaluate_unified_score`
        # already skips distance/m1 for - not silently reused as a mismatched value.
        detection_base, detection_base_n, feature_layer_keys = self._detection_base_from_parents(aggregation_refs, is_genesis_block)
        security_score, m1, m2, m3, raw_distance = self.evaluate_unified_score(
            model_x_weights=reconstructed_weights, parents_aggregated_weights=detection_base or {},
            actual_tested_acc=actual_tested_acc, parents_tested_acc=parents_tested_acc,
            claimed_class_acc=claimed_classes, evaluator_class_acc=eval_classes,
            is_genesis=is_genesis_block
        )
        # Feature-layers-only distance for the lazy gate specifically - see
        # `_detection_base_from_parents` docstring (classifier-layer
        # aggregation-function mismatch) and thesis-hub/docs/decisions.md,
        # 2026-09-10. `None` when there's no defined base, same convention as
        # `raw_distance`.
        lazy_check_distance = (
            self._calculate_relative_l2_distance(reconstructed_weights, detection_base, layer_keys=feature_layer_keys)
            if detection_base and feature_layer_keys else raw_distance
        )
        declared_coefficients = payload.get("aggregation_coefficients")
        declared_base = self._declared_base(declared_coefficients, aggregation_refs) if not is_genesis_block else None
        lazy_exact_only = bool(declared_base)
        if declared_base:
            lazy_check_distance = self._calculate_relative_l2_distance(reconstructed_weights, declared_base)
        # Diagnostic only (2026-09-10, thesis-hub/docs/decisions.md same date):
        # investigating why lazy nodes under `dual` selection almost never trip
        # the exact-zero gate (GATE-LAZY count near zero vs dozens under
        # `unified`, despite a node's own accuracy staying frozen the whole
        # run). Unconditional so the actual distance is visible even when it
        # clears BOTH gates - not just when one fires. Remove once the fix is
        # confirmed on a real Kaggle run.
        logger.debug(
            f"[DIAG-DISTANCE] proposer={proposer_id} raw_distance={raw_distance} "
            f"lazy_check_distance={lazy_check_distance} base_n={detection_base_n} "
            f"aggregation_refs_n={len(aggregation_refs)} parents_n={len(parent_hashes)} "
            f"is_genesis={is_genesis_block}"
        )

        # 🆕 Signal probe: record all trust-signal candidates for offline analysis
        if self.signal_probe is not None:
            self._record_probe_row(
                proposer_id=proposer_id,
                reconstructed_weights=reconstructed_weights,
                base_w=base_w if base_w else {},
                actual_tested_acc=actual_tested_acc,
                parents_tested_acc=parents_tested_acc,
                m3=m3,
                claimed_classes=claimed_classes,
                block_timestamp=block.get("timestamp", 0.0),
            )
        
        self._proposer_last_cid[proposer_id] = ipfs_cid

                # === Trust Score Filter (Additional layer from Signal Probe) ===
        if self.use_trust_score_filter:
            delta_acc = (actual_tested_acc - parents_tested_acc) / 100.0
            utility = max(0.0, min(1.0, (delta_acc + 0.3) / 0.6))  # Normalize to [0,1]
            class_entropy = self._calculate_class_entropy(claimed_classes)
            
            # Calculate composite trust score
            trust_score = (0.4 * utility) + (0.3 * class_entropy) + (0.3 * m3)
            
            if trust_score < self.trust_threshold:
                block["_validation_receipt"] = {
                    "status": "rejected",
                    "reason": "low_trust_score",
                    "trust_score": trust_score,
                    "utility": utility,
                    "entropy": class_entropy,
                    "m3": m3
                }
                
                self._flag_counter[proposer_id] = self._flag_counter.get(proposer_id, 0) + 1
                if self._flag_counter[proposer_id] >= self.MIN_PERSISTENCE:
                    self._flag_counter[proposer_id] = 0
                    self._publish_slash_evidence(
                        target_id=proposer_id,
                        reason="low_trust_score",
                        evidence={
                            "trust_score": trust_score,
                            "utility": utility,
                            "entropy": class_entropy,
                            "m3": m3,
                            "validator": self.node.node_id
                        }
                    )
                return False
            
        # 5. Anomaly Detection & Slashing
        if self.anomaly_mode == "gates":
            is_anomalous, anomaly_reason = self._detect_anomalies_gates(raw_distance, is_genesis_block, detection_base_n, lazy_check_distance, lazy_exact_only)
        else:
            is_anomalous, anomaly_reason = self._detect_anomalies(m1, m2, m3, actual_tested_acc, parents_tested_acc)
        is_alien = m3 < self.isolation_threshold
        
        if is_anomalous:
            inherited, max_parent_distance = self._check_inherited_contamination(
                anomaly_reason, raw_distance, aggregation_refs
            )
            if inherited:
                block["_validation_receipt"] = {
                    "status": "rejected",
                    "reason": anomaly_reason,
                    "raw_distance": raw_distance,
                    "inherited_contamination": True,
                }
                logger.info(
                    f"🧬 [INHERITED-CONTAMINATION] {proposer_id}'s distance={raw_distance:.4g} is "
                    f"explained by its own aggregation_refs (max parent distance={max_parent_distance:.4g}, "
                    f"tolerance={self.contamination_tolerance}) - rejecting block, NOT accusing proposer."
                )
                return False

            self._flag_counter[proposer_id] = self._flag_counter.get(proposer_id, 0) + 1
            block["_validation_receipt"] = {
                "status": "rejected", "reason": anomaly_reason, "raw_distance": raw_distance,
            }

            if self._flag_counter[proposer_id] >= self.MIN_PERSISTENCE:
                self._flag_counter[proposer_id] = 0
                # Only the S1-gates path (the live default, anomaly_mode="gates")
                # has a registered signal today. Legacy `_detect_anomalies`
                # reasons (poison_divergence/poison_severe_drop/poison_z_score)
                # intentionally fall through _validate_slash_evidence to
                # "cannot verify" until that dormant path also gets a registry
                # entry - see thesis-hub/docs/decisions.md, 2026-09-08.
                signal_id = "s1_relative_l2" if self.anomaly_mode == "gates" else None
                self._publish_slash_evidence(
                    target_id=proposer_id,
                    reason=anomaly_reason,
                    evidence={
                        "m1": m1, "m2": m2, "m3": m3,
                        "acc": actual_tested_acc,
                        "parent_acc": parents_tested_acc,
                        "reason": anomaly_reason,
                        "block_hash": block.get("hash", "unknown"),
                        "validator": self.node.node_id,
                    },
                    signal_id=signal_id,
                    block_ref={
                        "ipfs_cid": ipfs_cid,
                        "aggregation_refs": aggregation_refs,
                        "aggregation_coefficients": declared_coefficients,
                        "is_genesis": is_genesis_block,
                        "is_full_model": is_full_model,
                        "proposer": proposer_id,
                        "claimed_acc": payload.get("accuracy", 0.0),
                    },
                )
            else:
                logger.warning(
                    f"👁️ [SUSPICION] Flag {self._flag_counter[proposer_id]}/{self.MIN_PERSISTENCE} "
                    f"on {proposer_id} ({anomaly_reason}); awaiting persistence."
                )
            return False 
        
        # Safe block: Update history and assign weights
        if not is_genesis_block and not is_alien:
            self._update_m2_history(m2)
            self._record_accepted_distance(raw_distance, is_genesis_block, block.get("hash"))
        
        if is_alien:
            final_approval_weight = self.neutral_vote_weight * virtue
            logger.warning(f"⚖️ [ALERT] Node {proposer_id} is considered alien (M3: {m3:.4f}). Neutral approval weight applied.")
        else:
            final_approval_weight = security_score * m3 * virtue
            logger.info(f"✅ [VALIDATION] Node {proposer_id} passed deep inspection. Security Score: {security_score:.2f}, M1: {m1:.4f}, M2: {m2:.4f}, M3: {m3:.4f}, Virtue: {virtue:.2f}")
            
        # final_weight = self.neutral_vote_weight * virtue if is_alien else security_score * m3 * virtue * 100
        block["_validation_receipt"] = {
            "status": "approved", "weight": final_approval_weight, "raw_distance": raw_distance,
        }
        self._assign_approval_weights(payload, parent_hashes, final_approval_weight, proposer_id, reputation_module)
        
        return True

    def evaluate_unified_score(
        self, model_x_weights: Dict[str, torch.Tensor], parents_aggregated_weights: Dict[str, torch.Tensor],
        actual_tested_acc: float, parents_tested_acc: float, claimed_class_acc: Dict[int, float],
        evaluator_class_acc: Dict[int, float], is_genesis: bool = False
    ) -> Tuple[float, float, float, float, Optional[float]]:
        """
        Returns (security_score, m1, m2, m3, raw_distance).

        raw_distance is the un-transformed relative L2 and is None for a genesis
        block, where there is no base to measure against. It is returned rather
        than recomputed by the caller because it costs an O(P) pass over every
        tensor, and it must stay distinguishable from a real 0.0 (see the gates).
        """
        # An empty aggregated base is as undefined as a genesis block: relative_l2
        # would hit its `if not a or not b: return 0.0` guard and hand back a
        # perfect 0.0, which is indistinguishable from a lazy node republishing its
        # parent verbatim. `is_genesis` alone does NOT cover this - a block can have
        # parents whose aggregation still comes back empty, and in the plan-22 crash
        # check that produced 52 false lazy rejections in a run with no lazy nodes.
        distance = None
        if is_genesis or not parents_aggregated_weights:
            m1_novelty = 1.0 
        else:
            distance = self._calculate_relative_l2_distance(model_x_weights, parents_aggregated_weights)
            # 💡 1. Inverted Sigmoid: Penalize huge distances
            m1_novelty = 1.0 - (1.0 / (1.0 + math.exp(-self.novelty_steepness * (distance - self.novelty_midpoint))))
            logger.info(f"🔬 [TRACER-M1] Rel L2 Dist: {distance:.6e} -> M1 Novelty (Sigmoid): {m1_novelty:.4f}")
            
        m3_alignment = self._calculate_class_overlap(claimed_class_acc, evaluator_class_acc)
        is_exploitation = m3_alignment >= self.phase_detection_threshold
        current_steepness = self.strict_steepness if is_exploitation else self.lenient_steepness
        logger.info(f"🔬 [TRACER-M3] Class Overlap: {m3_alignment:.4f} | Phase: {'Exploitation' if is_exploitation else 'Exploration'} | Steepness: {current_steepness:.2f}")
        
        delta_acc_norm = (actual_tested_acc - parents_tested_acc) / 100.0
        if delta_acc_norm < 0:
            m2_utility = max(0.01, math.exp(current_steepness * delta_acc_norm))
            logger.info(f"🔬 [TRACER-M2] Accuracy Drop: {delta_acc_norm:.4f} -> M2 Utility (Exp): {m2_utility:.4f}")
        else:
            m2_utility = min(1.0, 1.0 + delta_acc_norm)
            logger.info(f"🔬 [TRACER-M2] Accuracy Gain: {delta_acc_norm:.4f} -> M2 Utility (Linear): {m2_utility:.4f}")

        w1, w2 = (self.w_plt_novelty, self.w_plt_utility) if is_exploitation else (self.w_exp_novelty, self.w_exp_utility)
        # 💡 2. Incorporate M3 as a Multiplicative Veto
        if is_exploitation:
            # در Exploitation، class overlap مهم است
            security_score = (m1_novelty ** w1) * (m2_utility ** w2) * m3_alignment * 100
        else:
            # در Exploration، Non-IID بودن طبیعی است → بدون veto
            security_score = (m1_novelty ** w1) * (m2_utility ** w2) * 100
            
        logger.info(f"🔬 [TRACER-SECURITY] Security Score: {security_score:.2f} | M1: {m1_novelty:.4f}, M2: {m2_utility:.4f}, M3: {m3_alignment:.4f}")
        
        return security_score, m1_novelty, m2_utility, m3_alignment, distance
    
    def _resolve_block_vertex(self, block_hash: str) -> Optional[dict]:
        """Look up a block by hash: local view first, global DAG as fallback.

        Mirrors the lookup pattern already used elsewhere in this file (see
        the aggregation-refs resolution above) rather than inventing a new
        one. By the time a block reaches `_record_accepted_distance`, its
        direct parents were already required to be locally resolvable to
        compute the S1 detection base, so this should not need the fallback
        in practice for depth computation.
        """
        if hasattr(self.node, "local_dag") and block_hash in self.node.local_dag.vertices:
            return self.node.local_dag.vertices.get(block_hash)
        if self.node and hasattr(self.node, "simulator") and hasattr(self.node.simulator, "dag"):
            return self.node.simulator.dag.vertices.get(block_hash)
        return None

    def _get_block_depth(self, block_hash: Optional[str]) -> Optional[int]:
        """
        Topological depth of a block: 0 for a genesis/parentless block, else
        1 + max(parent depths). Iterative (not recursive) to avoid Python's
        recursion limit on a long-running DAG. Returns None if the block or
        any ancestor needed to resolve its depth isn't locally resolvable --
        callers must treat that the same as an undefined distance (skip, do
        not guess), matching this gate's existing "fails toward a missed
        entry, not a wrong one" bias. See thesis-hub/docs/decisions.md,
        2026-09-17 (Path B).
        """
        if not block_hash:
            return None
        if block_hash in self._depth_cache:
            return self._depth_cache[block_hash]
        stack = [block_hash]
        while stack:
            h = stack[-1]
            if h in self._depth_cache:
                stack.pop()
                continue
            vertex = self._resolve_block_vertex(h)
            if vertex is None:
                self._depth_cache[h] = None
                stack.pop()
                continue
            parents = vertex.get("parents") or []
            if not parents:
                self._depth_cache[h] = 0
                stack.pop()
                continue
            unresolved = [p for p in parents if p not in self._depth_cache]
            if unresolved:
                stack.extend(unresolved)
                continue
            parent_depths = [self._depth_cache[p] for p in parents]
            self._depth_cache[h] = None if any(d is None for d in parent_depths) else 1 + max(parent_depths)
            stack.pop()
        return self._depth_cache.get(block_hash)

    def _current_gates(self) -> Tuple[float, float]:
        """
        Resolve the (low, high) distance gates for this validator.

        In adaptive mode the band is re-derived from this validator's own recent
        ACCEPTED distances, which is what makes the mechanism dataset-agnostic:
        the same code works on a different model or dataset without re-tuning a
        constant. Until enough history has accumulated, fall back to the values
        calibrated on MNIST honest-only data so cold start is not undefended.
        """
        if not self.gate_adaptive or len(self._distance_history) < self.gate_min_history:
            return self.gate_tau_low, self.gate_tau_high

        # Margins around the observed extremes — NOT percentiles. A percentile over a
        # bounded window is the wrong construction here: at history_size=200 a 0.1%
        # quantile degenerates to the window minimum, which sits well above the true
        # honest floor, so it clips honest blocks. Replayed over real plan-21 data the
        # quantile form produced 3.5-5.4% honest false positives, while margins around
        # the extremes produced 0.00% at identical 100% lazy/poisoning detection.
        #
        # A margin below the minimum is also the right shape for what is being
        # detected: a lazy free-rider republishes its parent verbatim, so its distance
        # is exactly 0.0 rather than merely small. The low gate only has to sit safely
        # under the honest floor, and the margin is what buys tolerance against an
        # attacker who adds a little noise to dodge an exact-zero test.
        distances = sorted(
            (item[2] for item in self._distance_history) if self.gate_dag_anchored_window
            else self._distance_history
        )
        # Outlier-sensitivity companion fix for Path B (thesis-hub/docs/decisions.md,
        # 2026-09-17): once there is more than the minimum required history, drop the
        # single most extreme accepted value at each end before taking min/max, so one
        # accepted-but-extreme honest distance cannot alone set the boundary -- the
        # same failure shape as the earlier GATE-POISON cold-start bug, this time
        # against a legitimately-accepted value rather than a skipped one. Deliberately
        # NOT a percentile (see above for why that construction is wrong here): exactly
        # the two extreme points are trimmed, not a fraction of the window.
        if len(distances) > self.gate_min_history:
            low_ref, high_ref = distances[1], distances[-2]
        else:
            low_ref, high_ref = distances[0], distances[-1]
        low = low_ref * self.gate_low_margin
        high = high_ref * self.gate_high_margin

        # RATCHET GUARD.
        # The history holds accepted blocks only, so it is a TRUNCATED sample of
        # the honest distance distribution, not a representative one. Every
        # rejection removes the current extreme, which lowers max(), which
        # tightens the high gate, which causes the next rejection: classic
        # selection-bias feedback. The offline replay could not see this because
        # it fed every honest row into the window regardless of the verdict.
        #
        # Measured in plan 22 group 1 (12 nodes, all honest): the band walked in
        # to tau_high 0.2048-0.2414 against a calibrated 0.2613 and tau_low
        # 0.0086-0.0169 against a calibrated 0.00274, and rejected 13 honest
        # blocks whose distances (0.2055-0.2521) were all UNDER the calibrated
        # ceiling. Two honest nodes were struck and slashed as a result.
        #
        # The fix keeps adaptation in the direction it was actually needed for —
        # a dataset whose honest band is WIDER than MNIST's still widens the
        # gates — while forbidding contraction below a band that was calibrated
        # on known-honest data. The trade-off is explicit: on a dataset whose
        # honest band is much narrower, the gates stay looser than optimal. That
        # fails toward missed detections rather than toward slashing honest
        # nodes, which is the correct direction for a mechanism whose penalty is
        # irreversible.
        if self.gate_clamp_to_bootstrap:
            low = min(low, self.gate_tau_low)
            high = max(high, self.gate_tau_high)
        return low, high

    def _has_adaptive_band(self) -> bool:
        """True once this validator has enough accepted history to place its own band."""
        return self.gate_adaptive and len(self._distance_history) >= self.gate_min_history

    def _in_gate_cold_start(self) -> bool:
        """
        True while this validator hasn't yet gate-evaluated gate_min_history blocks.

        Uses `_gate_blocks_seen` (every gate evaluation, any verdict), NOT
        `_distance_history` (accepted only). A 2026-09-08 fix skipped the high
        gate during cold-start so an early honest outlier wouldn't get slashed —
        but whatever the gate doesn't reject falls through and gets recorded as
        "accepted" (see `_record_accepted_distance`), so under an active
        mixed_attack run a poisoner's distance sailed through the same skip and
        permanently set max(_distance_history) to a poisoned value: GATE-POISON
        fired 132 times in the pre-guard run and ~0 times after, because the
        adaptive high threshold from then on was calibrated on poison itself.
        Cold-start must therefore also gate what enters the window, not just
        what gets slashed — and that gate can't be measured by window length,
        or a window that's rejecting everything never grows and cold-start
        never ends. See thesis-hub/docs/decisions.md, 2026-09-09.
        """
        return self.gate_adaptive and self._gate_blocks_seen < self.gate_min_history

    def _detect_anomalies_gates(
        self, distance: Optional[float], is_genesis: bool, base_size: Optional[int] = None,
        lazy_check_distance: Optional[float] = None,
        lazy_exact_only: bool = False,
    ) -> Tuple[bool, str]:
        """
        Two hard gates on the raw relative L2 distance.

        Returns (is_anomalous, reason). A genesis-parent block has no aggregated
        base, so the distance is undefined and BOTH gates must be skipped — note
        that `_calculate_relative_l2_distance` returns 0.0 for an empty base, which
        is indistinguishable from the lazy signature. Gating on that value flagged
        5.5% of honest blocks as lazy in the plan-21 data.

        `base_size` is how many parent blocks fed `_detection_base_from_parents`'s
        median/trimmed-mean (None if the caller predates this parameter - treated
        as "unknown, don't skip on this basis"). See the high-gate skip below for
        why this matters specifically under `dual` selection.

        `lazy_check_distance` (2026-09-10, thesis-hub/docs/decisions.md same
        date) is a SEPARATE, feature-layers-only distance used for the lazy
        gate specifically, defaulting to `distance` when not given (old
        callers). `aggregate()` (what a proposer actually trains from/copies)
        applies virtue-weighted averaging to classifier layers once its input
        set reaches `min_models_for_trimmed`, while the detection base always
        uses virtue-free median/trimmed-mean uniformly - the two are
        different quantities over identical inputs, so a verbatim-copy lazy
        node's FULL-model distance against the detection base is never
        exactly 0.0 once that threshold is crossed (confirmed: a
        frozen-accuracy lazy node produced the same nonzero 0.0672 at 10+
        independent validations). Feature layers alone stay virtue-free on
        both sides, so restricting the lazy check to them recovers the
        exact-zero signature without reintroducing virtue-weighting into
        detection (which the 2026-09-08 fix deliberately removed for
        reproducibility). The poison gate deliberately keeps comparing the
        FULL distance (`distance`, unchanged) - real poisoning magnitudes
        dwarf a classifier-only artifact either way, and restricting it would
        lose sensitivity to a poisoner that only manipulates classifier
        layers.
        """
        if lazy_check_distance is None or math.isnan(lazy_check_distance):
            lazy_check_distance = distance
        if is_genesis or distance is None or math.isnan(distance):
            return False, ""

        self._gate_blocks_seen += 1
        low, high = self._current_gates()

        # Reversed 2026-09-08: this gate used to fire unconditionally, on the belief
        # that poisoning sits so far above the honest band (709-1002 observed vs a
        # 0.26 bootstrap) that no regime shift makes the call marginal. That belief
        # was calibrated against the old virtue-weighted detection base. After
        # switching to the virtue-free median/trimmed-mean base the same day (see
        # `_detection_base_from_parents`), a real Kaggle run (group 3, mixed_attack)
        # produced an HONEST block at distance 689.5 - inside that exact "safe"
        # range - within the first ~43ms of the run, before any real mixing between
        # differently-trained Non-IID local models had happened. Gated the same way
        # the low gate already was: skip until this validator has its own adaptive
        # band (`gate_min_history` accepted distances), not a new threshold. The
        # accepted trade-off is the same one already made for the low gate: this
        # fails toward a missed detection during bootstrap rather than toward
        # slashing an honest node, which is the right direction for an irreversible
        # penalty. See thesis-hub/docs/decisions.md, 2026-09-08.
        #
        # Deliberately NOT `_has_adaptive_band()`/`_distance_history` length: that
        # helper is also False when `gate_adaptive=False` (pinned bootstrap,
        # "not mature yet" doesn't apply), AND, separately, a 2026-09-09 fix moved
        # cold-start off history length entirely for a second reason: history only
        # grows from blocks this same skip lets through, so gating cold-start on
        # it let a poisoner's own distance define "cold start over" - see
        # `_in_gate_cold_start` and thesis-hub/docs/decisions.md, 2026-09-09.
        # Thin-base skip (2026-09-10, thesis-hub/docs/decisions.md same date): a
        # median/trimmed-mean over too few parent blocks is not a representative
        # "normal" reference regardless of how much history THIS validator has
        # accumulated - the reference itself is degenerate, not the validator's
        # experience of it. Confirmed against a real Kaggle run (`mixed_dual`,
        # group 3): under `dual` selection the aggregation base is drawn from the
        # FINALIZED set, which fills far more slowly than `unified`'s tip pool
        # (thesis-hub/docs/decisions.md, 2026-09-07), so early on the base is
        # commonly built from just 1-2 blocks. Comparing a fast-evolving honest
        # proposer against that thin a sample produced a distinct, cheap,
        # exactly-repeating cluster of distances (~0.5-2) that cleared
        # `tau_high=0.2613` (calibrated on `unified`-like, tips-based honest data)
        # despite being nowhere near the genuine ~700-1000 poisoning cluster seen
        # in the same run. `min_models_for_trimmed` (already the aggregator's own
        # "is this quorum big enough to trust" threshold - see
        # `_aggregate_feature_layers`) is reused here rather than inventing a new
        # constant. Reusing the existing "fails toward a missed detection, not a
        # false slash" direction: skip, don't guess a per-mode threshold.
        #
        # Sanity-ceiling carve-out (2026-09-10, same date, thesis-hub/docs/decisions.md):
        # confirmed via a real Kaggle run (`mixed_dual`, seed_43) that an unconditional
        # thin-base skip lets a genuine poisoner through if it lands while the base is
        # still thin (distance=722.2, ~2800x tau_high, skipped and never flagged) - and
        # because `dual` mode's aggregation cites FINALIZED blocks, that one slip-through
        # doesn't get diluted, it compounds into every future round's aggregate for the
        # rest of the run. THIN_BASE_DISTANCE_SANITY_CEILING keeps the skip for genuinely
        # ambiguous thin-base noise (honest cluster topped out at 1.934) while still
        # flagging distances no honest noise could plausibly produce. Known residual gap:
        # an attacker landing just under the ceiling during a thin-base window is still
        # unflagged - not fixed here, see the same decisions.md entry for why that's
        # deferred.
        base_too_thin = (
            base_size is not None
            and base_size < self._detection_base_helper.min_models_for_trimmed
            and distance <= THIN_BASE_DISTANCE_SANITY_CEILING
        )
        cold_start_suppresses = self._in_gate_cold_start() and not (
            self.gate_cold_start_sanity_ceiling and distance > THIN_BASE_DISTANCE_SANITY_CEILING
        )
        high_gate_is_cold_start = cold_start_suppresses or base_too_thin
        if not high_gate_is_cold_start and distance > high:
            logger.warning(
                f"⛔ [GATE-POISON] distance={distance:.4g} > tau_high={high:.4g}"
            )
            return True, "poison_distance"
        if base_too_thin and distance > high:
            logger.info(
                f"👁️ [GATE-POISON-SKIP] distance={distance:.4g} > tau_high={high:.4g} "
                f"but base_size={base_size} < min_models_for_trimmed="
                f"{self._detection_base_helper.min_models_for_trimmed} - reference too thin to trust, not flagging."
            )

        # An exact zero is structural, not scale-dependent: the node re-published its
        # parent weights unchanged. Now that an undefined base yields None, this is an
        # unambiguous lazy signature in any regime. Uses `lazy_check_distance`
        # (feature-layers-only) rather than `distance` (full model) - see this
        # method's docstring for why the full-model distance is never exactly
        # 0.0 for a verbatim copy once the classifier layers cross into
        # virtue-weighted aggregation.
        if lazy_check_distance == 0.0:
            logger.warning("⛔ [GATE-LAZY] feature-layer distance is exactly 0 (verbatim republish)")
            return True, "lazy_distance"

        # The non-zero part of the low gate IS scale-dependent, so it only runs once
        # this validator has enough of its own accepted history to place it. Enforcing
        # the bootstrap value in a regime it was not calibrated for is self-defeating:
        # in the plan-22 crash check tau_low=0.00274 (fitted on Non-IID alpha=0.1,
        # 24 nodes) rejected honest blocks whose real distances were ~0.0002 in an IID
        # 8-node run, which stopped the history from ever filling, which stopped the
        # band from ever adapting away from the bad value.
        # lazy_exact_only: the distance is to the proposer's DECLARED base (W4).
        # tau_low was calibrated on the median-base distance, which carries the
        # median-vs-weighted-mean gap; the declared-base distance of an honest
        # update is far smaller (plan 46: 0.0013 late in training, below the
        # 0.00274 bootstrap tau_low -> one honest node slashed in 2 of 6
        # declared_on runs). Against the declared base only an exact zero is
        # meaningful, and it needs no scale.
        if not lazy_exact_only and self._has_adaptive_band() and lazy_check_distance < low:
            logger.warning(
                f"⛔ [GATE-LAZY] feature-layer distance={lazy_check_distance:.4g} < tau_low={low:.5g}"
            )
            return True, "lazy_distance"
        return False, ""

    def _record_accepted_distance(
        self, distance: Optional[float], is_genesis: bool, block_hash: Optional[str] = None,
    ) -> None:
        """
        Feed an accepted, well-defined distance into the adaptive band.

        Refuses entry while this validator is still in gate cold-start. Cold-start
        skips the high gate to avoid slashing an honest early outlier, so a block
        can reach here "accepted" without ever having been screened against the
        real threshold - most dangerously an active poisoner's own distance. Once
        such a value entered the window, max(_distance_history) permanently
        adopted it as the new "normal" for the life of the window (up to
        gate_history_size entries), which measured zero GATE-POISON detections for
        an entire mixed_attack run that should have produced ~132. See
        thesis-hub/docs/decisions.md, 2026-09-09.

        Path B (thesis-hub/docs/decisions.md, 2026-09-17): when
        `gate_dag_anchored_window` is set, the window holds (depth, hash,
        distance) triples and evicts the topologically OLDEST entry (lowest
        (depth, hash)) rather than the least-recently-appended one, so the
        window's contents depend only on which blocks this validator has
        accepted, not on the order network delivery happened to hand them
        over. `block_hash` must be resolvable in this validator's own local
        DAG view (see `_get_block_depth`); if depth can't be determined, the
        distance is dropped rather than guessed - same "fails toward a
        missed entry, not a wrong one" bias as the rest of this gate.
        """
        if is_genesis or distance is None or math.isnan(distance):
            return
        if self._in_gate_cold_start():
            return

        if self.gate_dag_anchored_window:
            depth = self._get_block_depth(block_hash)
            if depth is None:
                return
            self._distance_history.append((depth, block_hash or "", distance))
            if len(self._distance_history) > self.gate_history_size:
                oldest = min(self._distance_history, key=lambda item: (item[0], item[1]))
                self._distance_history.remove(oldest)
        else:
            self._distance_history.append(distance)
            if len(self._distance_history) > self.gate_history_size:
                self._distance_history.pop(0)

        # Emit the band periodically, not only when it fires. Plan 22 logged
        # tau only inside rejection warnings, so the band was observable at
        # exactly the moments it was already wrong and nowhere else — the
        # ratchet had to be reconstructed from the rejected values. This makes
        # "band over time vs the calibrated constant" directly plottable.
        if len(self._distance_history) % self.GATE_BAND_LOG_EVERY == 0:
            low, high = self._current_gates()
            window_distances = (
                [item[2] for item in self._distance_history] if self.gate_dag_anchored_window
                else self._distance_history
            )
            logger.info(
                f"📏 [GATE-BAND] n={len(self._distance_history)} "
                f"tau_low={low:.5g} tau_high={high:.5g} "
                f"win_min={min(window_distances):.5g} "
                f"win_max={max(window_distances):.5g}"
            )

    def _detect_anomalies(self, m1: float, m2: float,m3: float, acc: float, parent_acc: float) -> Tuple[bool, str]:
        """Detects malicious behavior using M1, M3 (Phase-Aware), and Z-score on M2."""        
        median_m2 = 0.0 
        mad = 0.0
        z_score = 0.0

        if m1 < 0.1 and m2 < self.divergence_m2_threshold:
            return True, "poison_divergence"
        
        # 💡 Phase-Aware Catastrophic Drop Trap
        # If M3 is low (Exploration Phase), we tolerate larger drops due to Non-IID data merging.
        is_exploration = m3 < self.phase_detection_threshold
        # Double the tolerance if exploring unfamiliar data distributions
        dynamic_drop_threshold = self.catastrophic_drop_threshold * (2.0 if is_exploration else 1.0)
        
        if (acc - parent_acc) / 100.0 < dynamic_drop_threshold:
            return True, "poison_severe_drop"
            
        if len(self.m2_score_history) >= 5:
            median_m2 = statistics.median(self.m2_score_history)
            mad = statistics.median([abs(x - median_m2) for x in self.m2_score_history])
            safe_mad = max(mad, 0.1) 
            z_score = (m2 - median_m2) / (1.4826 * safe_mad)
            dynamic_z_threshold = self.z_score_threshold * (2.0 if is_exploration else 1.0)
            logger.info(f"🔬 [TRACER-ZSCORE] Z-Score: {z_score:.4f} | M2: {m2:.4f}, Median: {median_m2:.4f}")
            if z_score < dynamic_z_threshold:
                logger.debug(
                    f"[ANOMALY_CHECK] validator={self.node.node_id} "
                    f"acc={acc:.4f} parent_acc={parent_acc:.4f} m1={m1:.4f} m2={m2:.4f} "
                    f"median_m2={median_m2:.4f} mad={mad:.4f} z_score={z_score:.4f} "
                    f"history_len={len(self.m2_score_history)}"
                )
                return True, "poison_z_score"
                
        return False, ""

    def _assign_approval_weights(self, payload: Dict[str, Any], parents: List[str], weight: float, proposer: str, rep_mod: Any):
        if "given_approval_weights" not in payload:
            payload["given_approval_weights"] = {}
            
        for p in parents:
            payload["given_approval_weights"][p] = weight
            
    def _lookup_parent_blocks(self, parent_hashes: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Resolve parent block hashes to their stored block dicts. None if any are missing locally."""
        parent_blocks = []
        for ph in parent_hashes:
            pb = self.node.local_dag.vertices.get(ph) if hasattr(self.node, 'local_dag') else None
            if not pb and hasattr(self.node.simulator, 'dag'):
                pb = self.node.simulator.dag.vertices.get(ph)
            if pb:
                parent_blocks.append(pb)
        if len(parent_blocks) < len(parent_hashes):
            return None
        return parent_blocks

    def _check_inherited_contamination(
        self, anomaly_reason: str, raw_distance: Optional[float], aggregation_refs: List[str]
    ) -> Tuple[bool, Optional[float]]:
        """
        One-hop exculpation for the poison gate (2026-09-16, thesis-hub/docs/
        decisions.md and research-direction.md item 14): an honest node that
        aggregates a not-yet-quarantined poisoned ancestor via its
        `aggregation_refs` carries that ancestor's distance signature and trips
        GATE-POISON just as hard as the actual poisoner. A real 3-seed Kaggle run
        (P4, hard workload) confirmed up to a 1:1 false:true accusation ratio.

        Before accusing the proposer, check whether its OWN direct
        aggregation_refs already carried a comparably large `raw_distance`
        (recorded in their own `_validation_receipt` the first time THIS
        validator processed them - same per-observer pattern as
        `_evidence_meets_quorum`, no replay needed). If so, this proposer added
        nothing new relative to its inputs and should not be personally accused
        - though its block is still rejected from further aggregation regardless.

        Deliberately single-hop: compares only against direct parents' own
        recorded distance, not a recursive walk back to the true origin of the
        contamination. Only applies to "poison_distance" - "lazy_distance" is
        about the proposer's OWN behavior (verbatim republish / frozen weights),
        not something inheritable from a parent's distance.
        """
        if anomaly_reason != "poison_distance" or raw_distance is None or not aggregation_refs:
            return False, None

        parent_blocks = self._lookup_parent_blocks(aggregation_refs)
        if not parent_blocks:
            return False, None

        parent_distances = [
            pb.get("_validation_receipt", {}).get("raw_distance") for pb in parent_blocks
        ]
        parent_distances = [d for d in parent_distances if d is not None]
        if not parent_distances:
            return False, None

        max_parent_distance = max(parent_distances)
        excess = raw_distance - max_parent_distance
        return excess <= self.contamination_tolerance, max_parent_distance

    def _aggregate_parents_if_needed(self, parent_hashes: List[str], is_genesis: bool, proposer_id: str) -> Optional[Dict[str, torch.Tensor]]:
        if is_genesis: return {}
        parent_blocks = self._lookup_parent_blocks(parent_hashes)
        if parent_blocks is None:
            logger.warning(f"[VALIDATION] Soft reject {proposer_id}: Missing parents.")
            return None
        return self.node.aggregator.aggregate(parent_blocks, node=self.node) if hasattr(self.node, 'aggregator') else {}

    def _detection_base_from_parents(self, parent_hashes: List[str], is_genesis: bool) -> Tuple[Optional[Dict[str, torch.Tensor]], int, List[str]]:
        """
        Virtue-free, fully deterministic reference for the S1 distance signal
        ONLY - not for reconstructing a block's own weights. See the
        `_detection_base_helper` comment in `__init__` for why
        `_resolve_aggregation_inputs`/`_filter_weight_space_anomalies` are
        deliberately not reused here. Returns `({}, 0, [])` for genesis (no
        base exists), `(None, 0, [])` if a referenced parent cannot be fetched
        locally (caller should treat this like an undefined distance, not
        silently fall back to the unstable virtue-weighted base).

        Also returns how many parent blocks actually contributed to the base
        (`n_valid`). Under `dual` selection, this base is built from the
        FINALIZED set, which fills far more slowly than the tip pool `unified`
        uses (see thesis-hub/docs/decisions.md, 2026-09-07) - so early in a
        run it is common for only one or two blocks to be available. The
        median/trimmed-mean over that few a sample is not a representative
        "normal" reference, and comparing a fast-evolving honest proposer's
        weights against it produces distances far above `tau_high` that have
        nothing to do with poisoning - confirmed 2026-09-10 against a real
        Kaggle run (group 3, mixed_attack, `mixed_dual`): 87.5% of honest
        nodes struck at least once, GATE-POISON distances clearly bimodal
        (a small, cheap, exactly-repeating cluster around 0.5-2 alongside the
        genuine ~700-1000 poisoning cluster). `n_valid` lets the gate (see
        `_detect_anomalies_gates`) tell the two apart instead of trusting
        `tau_high` at a scale it was never calibrated for.

        Finally returns `feature_layer_keys` (via the same, virtue-free
        `_detect_layer_types` heuristic `aggregate()` itself uses to route
        feature vs classifier layers). A genuinely lazy proposer's classifier
        layers get VIRTUE-WEIGHTED averaging in `aggregate()` whenever its
        input set reaches `min_models_for_trimmed`, while this base always
        uses median/trimmed-mean uniformly - the two are not the same
        quantity even over identical inputs, so a verbatim-copy lazy node
        never lands on exactly 0.0 distance against the FULL base. Confirmed
        2026-09-10 (group 3, mixed_attack, `mixed_dual`): a frozen-accuracy
        (15.42% all run) lazy node produced the same nonzero distance
        (0.0672) at 10+ independent validations - deterministic, not noise.
        Feature layers alone stay virtue-free in both places, so the lazy
        gate uses only those (see `_detect_anomalies_gates`); the poison gate
        keeps comparing the full distance, since real poisoning magnitudes
        (~700+) dwarf a classifier-only artifact either way and restricting
        it would lose sensitivity to classifier-only poisoning.
        """
        if is_genesis:
            return {}, 0, []
        parent_blocks = self._lookup_parent_blocks(parent_hashes)
        if parent_blocks is None:
            return None, 0, []

        helper = self._detection_base_helper
        valid_payloads = []
        for pb in parent_blocks:
            weights = helper._lazy_fetch_weights(pb, self.node)
            if not weights:
                continue
            inner = pb.get("payload", pb)
            node_id = (
                pb.get("author") or pb.get("proposer")
                or inner.get("author") or inner.get("proposer") or "unknown"
            )
            valid_payloads.append({"node_id": node_id, "weights": weights, "accuracy": 0.0, "samples": 1})

        if not valid_payloads:
            return None, 0, []

        all_layers = list(valid_payloads[0]["weights"].keys())
        feature_layers, _classifier_layers = helper._detect_layer_types(valid_payloads)
        try:
            base = helper._aggregate_feature_layers(valid_payloads, all_layers)
            return base, len(valid_payloads), feature_layers
        except Exception as e:
            logger.error(f"[DETECTION-BASE] Median/trimmed-mean computation failed: {e}")
            return None, 0, []

    def _lookup_block_any(self, block_hash: Optional[str]) -> Optional[Dict[str, Any]]:
        """A block header from any store this node holds (L2 DAG, local DAG, shared DAG)."""
        if not block_hash or self.node is None:
            return None
        for store in (
            getattr(getattr(self.node, "l2_strategy", None), "blocks", None),
            getattr(getattr(self.node, "local_dag", None), "vertices", None),
            getattr(getattr(getattr(self.node, "simulator", None), "dag", None), "vertices", None),
        ):
            if isinstance(store, dict) and block_hash in store:
                return store[block_hash]
        return None

    @staticmethod
    def _header_fields(block: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float, List[str], Any]:
        payload = block.get("payload", {}) if isinstance(block.get("payload"), dict) else {}
        cid = payload.get("ipfs_cid") or block.get("ipfs_cid")
        author = block.get("proposer") or block.get("author") or payload.get("author")
        ts = float(block.get("timestamp", 0.0) or 0.0)
        refs = payload.get("aggregation_refs") or block.get("aggregation_refs") or block.get("parents") or payload.get("parents") or []
        coefficients = payload.get("aggregation_coefficients")
        return cid, author, ts, list(refs), coefficients

    def _cid_theft_proven(self, accused: Optional[Dict[str, Any]], victim: Optional[Dict[str, Any]]) -> bool:
        """The four conditions, from headers only. Conservative: any doubt -> False."""
        if not isinstance(accused, dict) or not isinstance(victim, dict):
            return False
        cid_a, author_a, ts_a, refs_a, coef_a = self._header_fields(accused)
        cid_v, author_v, ts_v, _, _ = self._header_fields(victim)
        if not cid_a or cid_a != cid_v:                       # (1)
            return False
        if not ts_v < ts_a:                                   # (2)
            return False
        if not author_a or not author_v or author_a == author_v:  # (3)
            return False
        # Cold-start carve-out: no refs, or only genesis refs (no CID). Every
        # node starts from the same initial model, so two nodes that publish it
        # untrained share a CID without either having done the other's work -
        # smoke test 2026-09-26: two lazy nodes on genesis, 28ms apart.
        ref_blocks = []
        for h in refs_a:
            ref = self._lookup_block_any(h)
            if ref is None:
                return False
            ref_blocks.append((h, ref))
        if not any(self._header_fields(ref)[0] for _, ref in ref_blocks):
            return False
        # (4) Refs carrying the same content as the victim are the only way the
        # accused's own aggregation base could equal the victim. If there are
        # none, a match cannot be a republish. If there are some, the declared
        # coefficients must show weight on at least one ref with different
        # content; a missing/unreadable declaration is treated as a republish.
        same_content = [h for h, ref in ref_blocks if self._header_fields(ref)[0] == cid_a]
        if not same_content:
            return True
        if not isinstance(coef_a, list):
            return False
        try:
            weight_elsewhere = sum(float(c) for h, c in coef_a if h not in same_content)
        except (TypeError, ValueError):
            return False
        return weight_elsewhere > 1e-9

    def _check_cid_theft(self, block: Dict[str, Any], proposer_id: str) -> bool:
        """
        Header-level stealing check, run for every validated block before tier
        routing. Returns True when the INCOMING block is the thief (caller
        rejects it). If an older block with the same CID arrives after the copy,
        evidence goes against the already-seen copy instead - order is by ready
        time, never by gossip arrival.
        """
        cid, _, ts, _, _ = self._header_fields(block)
        block_hash = block.get("hash")
        if not cid or not block_hash:
            return False
        seen_hash = self._cid_index.get(cid)
        seen = self._lookup_block_any(seen_hash) if seen_hash and seen_hash != block_hash else None
        if seen is None:
            self._cid_index[cid] = block_hash
            return False
        seen_ts = self._header_fields(seen)[2]
        if seen_ts <= ts:
            accused, victim = block, seen
        else:
            accused, victim = seen, block
            self._cid_index[cid] = block_hash
        if not self._cid_theft_proven(accused, victim):
            return False
        accused_author = self._header_fields(accused)[1]
        victim_author = self._header_fields(victim)[1]
        logger.critical(
            f"🕵️ [STEAL-CID] {accused_author} published {victim_author}'s model verbatim "
            f"(cid {cid[:12]}, victim ready {self._header_fields(victim)[2]:.3f}s < accused "
            f"{self._header_fields(accused)[2]:.3f}s)"
        )
        self._publish_slash_evidence(
            target_id=accused_author,
            reason="steal",
            evidence={"victim": victim_author, "cid": cid, "validator": self.node.node_id},
            signal_id="steal_cid_match",
            block_ref={"accused_hash": accused.get("hash"), "victim_hash": victim.get("hash"),
                       "proposer": accused_author},
        )
        if accused is block:
            block["_validation_receipt"] = {"status": "rejected", "reason": "steal"}
            return True
        return False

    def _declared_base(self, coefficients: Any, aggregation_refs: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Rebuild the exact base the proposer says it aggregated (W4): sum_j c_j * W_j
        over all layers, in the declared order (same summation order as
        FarabiProxAggregator, so an unmodified republish is bit-identical).
        None - caller keeps the median reference - when the flag is off or the
        declaration is malformed: not a list of [hash, coef], a hash outside
        aggregation_refs, a negative coefficient, or a sum away from 1.
        """
        if not self.declared_aggregation_base or not coefficients or not isinstance(coefficients, list):
            return None
        refs = set(aggregation_refs or [])
        pairs = []
        for item in coefficients:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None
            h, c = item[0], item[1]
            try:
                c = float(c)
            except (TypeError, ValueError):
                return None
            if h not in refs or c < 0.0 or math.isnan(c):
                return None
            pairs.append((h, c))
        if abs(sum(c for _, c in pairs) - 1.0) > 1e-6:
            return None
        blocks = self._lookup_parent_blocks([h for h, _ in pairs])
        if blocks is None:
            return None
        helper = self._detection_base_helper
        base: Dict[str, torch.Tensor] = {}
        for (h, c), pb in zip(pairs, blocks):
            weights = helper._lazy_fetch_weights(pb, self.node)
            if not weights:
                return None
            for k, v in weights.items():
                scaled = v.float() * c
                base[k] = scaled if k not in base else base[k] + scaled
        return base

    def _update_m2_history(self, m2: float):
        self.m2_score_history.append(m2)
        logger.debug(
            f"[M2_HISTORY_UPDATE] validator={self.node.node_id}  m2={m2:.4f}"
        )

        if len(self.m2_score_history) > self.history_window_size:
            self.m2_score_history.pop(0)
            logger.info(f"🔬 [TRACER-M2-HISTORY] Sliding window updated. Current M2 history: {self.m2_score_history}")

    # --- Helper Data & Math Methods ---
    def _fetch_data_from_ipfs(self, cid: str) -> Optional[Dict[str, torch.Tensor]]:
        if not cid: return None
        if cid in self.local_hot_cache: return self.local_hot_cache[cid]
        
        mock_ipfs = getattr(self.node.simulator, 'mock_ipfs', MockIPFS()) if hasattr(self.node, 'simulator') else None
        try:
            data_weights, data_size_bytes = download_model_from_ipfs(cid, self.storage_config, mock_ipfs)
            if hasattr(self.node, 'telemetry'):
                cap = getattr(self.node, 'capacity', 1.0)
                delay = getattr(self.node, 'delay_multiplier', 1.0)
                bw = (self.base_bw_mbps * cap) / max(0.01, delay)
                sim_time = (data_size_bytes / (1024 * 1024) / (bw / 8.0)) + self.network_overhead_sec if bw > 0 else 0.0
                self.node.telemetry.register_network_delay(sim_time)
                self.node.telemetry.register_traffic(data_size_bytes, channel="ipfs")
            self.local_hot_cache[cid] = data_weights
            return data_weights
        except Exception as e:
            logger.error(f"IPFS Fetch Failed: {e}")
            return None

    def _reconstruct_full_model(self, base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        if not base or not delta: return None
        try:
            return {k: base[k] + delta[k] for k in delta.keys() if k in base}
        except Exception as e:
            logger.error(f"Model Reconstruction Failed: {e}")
            return None

    def _verify_frankenstein_trap(self, weights: Dict[str, torch.Tensor], claimed_fp: float) -> bool:
        if not weights: return False
        try:
            last_tensor = weights[list(weights.keys())[-1]]
            actual_fp = float(torch.sum(torch.abs(last_tensor)).item())
            diff = abs(actual_fp - claimed_fp)
            return diff <= self.frankenstein_tolerance
        except Exception as e:
            logger.error(f"Frankenstein Trap Verification Failed: {e}")
            return False

    def _calculate_relative_l2_distance(
        self, w_new: Dict[str, torch.Tensor], w_old: Dict[str, torch.Tensor],
        layer_keys: Optional[List[str]] = None,
    ) -> float:
        """
        `layer_keys`, when given, restricts the sum to those layer names only
        (still intersected with keys present in both dicts). Used to compute
        a feature-layers-only distance for the lazy gate - see
        `_detect_anomalies_gates` for why the full-model distance isn't safe
        for that specific check.
        """
        if not w_new or not w_old: return 0.0
        keys = w_new.keys() if layer_keys is None else layer_keys
        delta_norm_sq = sum((w_new[k] - w_old[k]).norm().item() ** 2 for k in keys if k in w_new and k in w_old)
        base_keys = w_old.keys() if layer_keys is None else layer_keys
        base_norm_sq = sum(w_old[k].norm().item() ** 2 for k in base_keys if k in w_old)
        return math.sqrt(delta_norm_sq) / (math.sqrt(base_norm_sq) + self.epsilon)

    def _calculate_class_overlap(self, dist1: Dict[int, float], dist2: Dict[int, float]) -> float:
        """
        Calculates Intersection-based Class Overlap.
        Prevents unfairly penalizing nodes for naturally lacking certain classes in a Non-IID setup.
        """
        if not dist1 or not dist2: 
            return 0.0
            
        common_classes = set(dist1.keys()).intersection(set(dist2.keys()))
        
        if not common_classes:
            # 💡 رویکرد آشتی‌جویانه: اگر هیچ اشتراکی ندارند، به جای صفر، نمره خنثی بده تا مدل‌های جدید فرصت ارزیابی پیدا کنند
            return 0.5 
            
        overlap = sum(min(dist1[cls], dist2[cls]) for cls in common_classes)
        
        # نرمال‌سازی بر اساس تعداد کلاس‌های مشترک (تقسیم بر 100 برای مقیاس 0 تا 1)
        return min(max(overlap / (len(common_classes) * 100.0), 0.0), 1.0)
    
    # --- Signal registry: shared source of truth for detection AND verification ---

    def _build_signal_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        One entry per independently-recomputable signal. `materials` fetches the
        raw inputs (never trusts a self-reported derived value); `compute` is the
        exact function the detector itself used; `gate` re-applies the decision
        rule against THIS verifier's own local state (its own adaptive band, its
        own fingerprint registry) rather than the publisher's claim.

        Adding a new detector = adding one entry here. Both detection and peer
        verification read from it, so they cannot drift apart the way
        `_detect_anomalies_gates` and the old string-matched
        `_validate_slash_evidence` did.
        """
        return {
            "s1_relative_l2": {
                "materials": self._materials_for_distance_signal,
                # Bundles n_valid and a feature-layers-only distance alongside
                # the full-model distance so `_gate_s1_relative_l2` can apply
                # the same thin-base skip AND the same lazy-gate feature-layer
                # restriction the detector used - independently recomputed
                # here, not trusted from the (self-reported, untrusted)
                # evidence dict. See `_detection_base_from_parents` and
                # `_detect_anomalies_gates`, 2026-09-10.
                "compute": lambda w_new, w_old, n_valid, feature_layer_keys, w_declared=None: (
                    self._calculate_relative_l2_distance(w_new, w_old),
                    n_valid,
                    self._calculate_relative_l2_distance(w_new, w_declared) if w_declared
                    else (self._calculate_relative_l2_distance(w_new, w_old, layer_keys=feature_layer_keys) if feature_layer_keys else None),
                    bool(w_declared),
                ),
                "gate": self._gate_s1_relative_l2,
            },
            "steal_cid_match": {
                "materials": lambda block_ref: (
                    {"accused": self._lookup_block_any(block_ref.get("accused_hash")),
                     "victim": self._lookup_block_any(block_ref.get("victim_hash"))}
                    if self._lookup_block_any(block_ref.get("accused_hash")) is not None
                    and self._lookup_block_any(block_ref.get("victim_hash")) is not None else None
                ),
                "compute": lambda accused, victim: self._cid_theft_proven(accused, victim),
                "gate": lambda proven, evidence: bool(proven)
                    and self._header_fields(self._lookup_block_any(evidence.get("block_ref", {}).get("accused_hash")) or {})[1] == evidence.get("target"),
            },
            "steal_hash_collision": {
                "materials": self._materials_for_weights_only,
                "compute": lambda weights: self._strong_model_hash(weights),
                "gate": self._gate_steal_hash_collision,
            },
            "frankenstein_checksum": {
                "materials": self._materials_for_weights_only,
                "compute": lambda weights: self._frankenstein_fingerprint(weights),
                "gate": self._gate_frankenstein_checksum,
            },
            "metadata_commitment_mismatch": {
                "materials": self._materials_for_metadata_commitment,
                "compute": lambda class_accuracies, accuracy: generate_metadata_commitment(class_accuracies, accuracy),
                "gate": self._gate_metadata_commitment,
            },
            "suspicious_metadata": {
                "materials": lambda block_ref: {"claimed_acc": block_ref.get("claimed_acc")},
                "compute": lambda claimed_acc: claimed_acc,
                "gate": lambda value, evidence: value is not None and value < 0.1,
            },
        }

    def _materials_for_distance_signal(self, block_ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Re-fetch the disputed block's own weights (`w_new`) and independently
        recompute the comparison base (`w_old`) via `_detection_base_from_parents`
        - NOT `_aggregate_parents_if_needed`, which is virtue-weighted and,
        under `dual` selection, re-picks whatever is currently finalized. Using
        that here is exactly the bug this whole registry was built to avoid:
        every verifier recomputing a DIFFERENT `w_old` than the original
        detector saw, purely because time/Virtue moved on between detection
        and verification. `_aggregate_parents_if_needed` is still used ONLY to
        reconstruct `w_new` itself when a block is delta-encoded (dormant path
        today - see the `_detection_base_helper` comment in `__init__`).
        """
        ipfs_cid = block_ref.get("ipfs_cid")
        is_genesis = bool(block_ref.get("is_genesis", False))
        if not ipfs_cid:
            return None
        downloaded = self._fetch_data_from_ipfs(ipfs_cid)
        if not downloaded:
            return None
        aggregation_refs = block_ref.get("aggregation_refs") or []
        if block_ref.get("is_full_model", True):
            reconstructed = downloaded
        else:
            reconstruction_base = self._aggregate_parents_if_needed(aggregation_refs, is_genesis, block_ref.get("proposer", "unknown"))
            if reconstruction_base is None and not is_genesis:
                return None
            reconstructed = self._reconstruct_full_model(reconstruction_base or {}, downloaded)
        if not reconstructed:
            return None

        detection_base, detection_base_n, feature_layer_keys = self._detection_base_from_parents(aggregation_refs, is_genesis)
        if detection_base is None:
            return None
        declared_base = None if is_genesis else self._declared_base(block_ref.get("aggregation_coefficients"), aggregation_refs)
        return {
            "w_new": reconstructed, "w_old": detection_base, "n_valid": detection_base_n,
            "feature_layer_keys": feature_layer_keys, "w_declared": declared_base,
        }

    def _materials_for_weights_only(self, block_ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Re-fetch + re-reconstruct the disputed block's full weights (no base needed)."""
        materials = self._materials_for_distance_signal(block_ref)
        if materials is None:
            return None
        return {"weights": materials["w_new"]}

    def _materials_for_metadata_commitment(self, block_ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Metadata-commitment checks are about internal consistency of the
        target's OWN claims (does its claimed hash match its claimed numbers),
        not ground truth - so reading the claims from block_ref (populated from
        the actual disputed block, not the evidence publisher's summary) is the
        right level of trust, no IPFS fetch needed.
        """
        class_accuracies = block_ref.get("class_accuracies")
        accuracy = block_ref.get("accuracy")
        if class_accuracies is None or accuracy is None:
            return None
        return {"class_accuracies": class_accuracies, "accuracy": accuracy}

    def _frankenstein_fingerprint(self, weights: Dict[str, torch.Tensor]) -> Optional[float]:
        if not weights:
            return None
        try:
            last_tensor = weights[list(weights.keys())[-1]]
            return float(torch.sum(torch.abs(last_tensor)).item())
        except Exception as e:
            logger.error(f"[EVIDENCE] Frankenstein fingerprint recompute failed: {e}")
            return None

    def _gate_s1_relative_l2(self, recomputed: Optional[Tuple[Optional[float], int, Optional[float]]], evidence: Dict[str, Any]) -> bool:
        """Independently re-run the SAME two-gate rule the detector used, on a value THIS verifier computed itself."""
        if recomputed is None:
            return False
        recomputed_distance, recomputed_base_n, recomputed_lazy_distance = recomputed[:3]
        lazy_exact_only = bool(recomputed[3]) if len(recomputed) > 3 else False
        if recomputed_distance is None:
            return False
        is_genesis = bool(evidence.get("block_ref", {}).get("is_genesis", False))
        is_anomalous, actual_reason = self._detect_anomalies_gates(
            recomputed_distance, is_genesis, recomputed_base_n, recomputed_lazy_distance, lazy_exact_only
        )
        # Require the claimed verdict to match what recomputation ACTUALLY finds -
        # closes the old bug where "poison_distance" evidence was verified against
        # an unrelated accuracy-drop criterion instead of the distance itself.
        return is_anomalous and actual_reason == evidence.get("verdict_label")

    def _gate_steal_hash_collision(self, recomputed_hash: Optional[str], evidence: Dict[str, Any]) -> bool:
        if not recomputed_hash:
            return False
        target = evidence.get("target")
        victim = self._fp_registry.get(recomputed_hash)
        return victim is not None and victim != target

    def _gate_frankenstein_checksum(self, recomputed_fp: Optional[float], evidence: Dict[str, Any]) -> bool:
        claimed_fp = evidence.get("block_ref", {}).get("claimed_fingerprint")
        if claimed_fp is None or recomputed_fp is None:
            return False
        return abs(recomputed_fp - claimed_fp) > self.frankenstein_tolerance

    def _gate_metadata_commitment(self, recomputed_commitment: Optional[str], evidence: Dict[str, Any]) -> bool:
        claimed_commitment = evidence.get("block_ref", {}).get("claimed_commitment")
        return bool(claimed_commitment) and recomputed_commitment != claimed_commitment

    def _publish_slash_evidence(
        self,
        target_id: str,
        reason: str,
        evidence: Dict[str, Any],
        signal_id: Optional[str] = None,
        block_ref: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates and publishes a SlashEvidenceBlock in the DAG.
        Other nodes will approve/reject it via On-Tangle Voting.
        No immediate slashing occurs.

        `signal_id` + `block_ref` are what peers use to INDEPENDENTLY recompute
        and verify (see `_signal_registry`/`_validate_slash_evidence`); `reason`
        and `evidence` remain for logging/backward-compat display only and are
        never trusted as the basis for a vote.
        """
        import hashlib

        current_time = getattr(self.node.simulator, 'current_time', 0.0) if self.node else 0.0
        evidence_with_reason = dict(evidence)
        evidence_with_reason["reason"] = reason
        # `_process_slash_evidence` only forwards this inner dict (not the outer
        # block) to `_validate_slash_evidence` - signal_id/block_ref/target MUST
        # live here, not only on the outer slash_evidence block below, or every
        # peer's independent verification silently sees signal_id=None again.
        evidence_with_reason["signal_id"] = signal_id
        evidence_with_reason["verdict_label"] = reason
        evidence_with_reason["block_ref"] = block_ref or {}
        evidence_with_reason["target"] = target_id
        evidence_hash = hashlib.sha256(f"slash_{target_id}_{reason}".encode()).hexdigest()

        publisher_virtue = 1.0
        if self.node and hasattr(self.node, 'reputation'):
            publisher_virtue = self.node.reputation.get_virtue_score(self.node.node_id)


        slash_evidence = {
            "type": SLASH_EVIDENCE_TYPE,
            "hash": evidence_hash,
            "timestamp": current_time,
            "proposer": self.node.node_id,
            "target": target_id,
            "reason": reason,
            "signal_id": signal_id,
            "verdict_label": reason,
            "block_ref": block_ref or {},
            "evidence": evidence_with_reason,
            "approvals": {self.node.node_id},  # Proposer auto-approves
            "approval_weight": 0.0,
            "approval_weight": publisher_virtue,
        }
        
        # Add to L2 DAG if available
        #
        # `evidence_hash` is a pure function of (target, reason), so every
        # re-detection of the same fault maps onto ONE evidence id and
        # `add_slash_evidence` dedupes it away. Logging unconditionally below
        # therefore reported publication volume, not evidence volume: in plan 23
        # group 3 / seed_43 `dual` this printed 1718 "Published evidence" lines
        # for 53 actually-stored evidence items, which made the run look like an
        # evidence flood diluting the vote quorum. It was not - the real cause
        # was the single-observer quorum check in `_check_slash_confirmation`
        # (fixed 2026-09-12). Log the deduped outcome so the count in the log
        # means what a reader assumes it means.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        is_new = True
        l2 = getattr(self.node, 'l2_strategy', None) if self.node else None
        if l2 is not None and hasattr(l2, 'add_block'):
            pending = getattr(l2, 'pending_slash_evidence', {})
            is_new = evidence_hash not in pending
            l2.add_block(
                slash_evidence,
                simulator=getattr(self.node, 'simulator', None)
            )
        if is_new:
            logger.warning(
                f"🗳️ [SLASH_EVIDENCE] Published evidence against {target_id} "
                f"(reason: {reason}). Awaiting on-tangle approval."
            )
        else:
            logger.debug(
                f"🗳️ [SLASH_EVIDENCE] Re-detected {reason} against {target_id}; "
                f"evidence {evidence_hash[:8]} already pending, not republished."
            )
        
    def _process_slash_evidence(self, evidence_block: Dict[str, Any]) -> bool:
        """
        Processes a received SlashEvidenceBlock.
        Independently validates the evidence using EXISTING consensus thresholds.
        If convinced, casts an approval vote via the L2 DAG.
        
        Returns True so BaseNode adds it to local_dag (protocol-agnostic behavior).
        """
        target = evidence_block.get("target")
        proposer = evidence_block.get("proposer")
        evidence = evidence_block.get("evidence", {})
        evidence_id = evidence_block.get("hash")

        if not target or not evidence_id:
            logger.warning("[SLASH_EVIDENCE] Missing target or hash. Rejecting.")
            return False

        # Don't approve my own evidence twice
        if proposer == self.node.node_id:
            return True

        # ✅ Independent validation using EXISTING thresholds
        is_valid = self._validate_slash_evidence(evidence)

        if is_valid:
            # Cast approval vote via L2 DAG
            voter_virtue = 1.0
            if hasattr(self.node, 'reputation'):
                voter_virtue = self.node.reputation.get_virtue_score(self.node.node_id)

            if hasattr(self.node, 'l2_strategy') and hasattr(self.node.l2_strategy, 'approve_slash_evidence'):
                self.node.l2_strategy.approve_slash_evidence(
                    evidence_id=evidence_id,
                    voter_id=self.node.node_id,
                    voter_virtue=voter_virtue
                )

            logger.info(
                f"✅ [SLASH_VOTE] Node {self.node.node_id} approved evidence "
                f"against {target} (virtue: {voter_virtue:.2f})"
            )
        else:
            logger.info(
                f"⏭️ [SLASH_VOTE] Node {self.node.node_id} found evidence "
                f"against {target} insufficient. Abstaining."
            )

        # Return False: BaseNode should NOT process this as a regular transaction
        # The slash evidence is already stored in l2_strategy.pending_slash_evidenc
        return False


    def _validate_slash_evidence(self, evidence: Dict[str, Any]) -> bool:
        """
        Independently validates slash evidence by RECOMPUTING the underlying
        signal from raw materials (IPFS-fetched weights, re-run aggregation),
        never by trusting the publisher's self-reported numbers or by
        pattern-matching the `reason` string.

        This is signal-driven, not attack-type-driven: adding a new detector
        means adding one entry to `_signal_registry` (see `_build_signal_registry`)
        - both detection and this verification read the same compute+gate pair,
        so they cannot silently drift apart. It generalizes to any NEW COMBINATION
        of signals already in the registry; it does NOT grant detection of an
        attack that affects no registered signal at all - no evidence scheme can
        verify a measurement nobody takes. See thesis-hub/docs/decisions.md,
        2026-09-08, for the full derivation (including why evidence-verification
        sampling was considered and deferred, and why min_voters was left at 3).

        Legacy `reason` strings not backed by a `signal_id` (e.g. evidence
        published by `_detect_anomalies`, the `anomaly_mode="legacy"` path, or
        `use_trust_score_filter`) fall through to `return False` - an explicit,
        loggable "cannot independently verify this", not a silent mismatch.
        """
        signal_id = evidence.get("signal_id")
        spec = self._signal_registry.get(signal_id)
        if spec is None:
            logger.info(
                f"[EVIDENCE] No registered signal for signal_id={signal_id!r} "
                f"(reason={evidence.get('reason')!r}) - cannot independently "
                f"verify, abstaining."
            )
            return False

        block_ref = evidence.get("block_ref", {})
        try:
            materials = spec["materials"](block_ref)
        except Exception as e:
            logger.error(f"[EVIDENCE] Materials fetch failed for signal_id={signal_id!r}: {e}")
            return False
        if materials is None:
            logger.info(f"[EVIDENCE] Could not independently fetch materials for signal_id={signal_id!r} - abstaining.")
            return False

        try:
            recomputed_value = spec["compute"](**materials)
        except Exception as e:
            logger.error(f"[EVIDENCE] Recompute failed for signal_id={signal_id!r}: {e}")
            return False

        return bool(spec["gate"](recomputed_value, evidence))
    
    def _strong_model_hash(self, weights: Dict[str, torch.Tensor]) -> str:
        """
        هش قوی و پایدار برای تشخیص Plagiarism.
        گردکردن تا ۴ رقم اعشار، نویز عددی ناچیز را نادیده می‌گیرد
        ولی کپی دقیق را با احتمال ~۱ شناسایی می‌کند.
        """
        import hashlib
        h = hashlib.sha256()
        for k in sorted(weights.keys()):
            t = weights[k].detach().cpu().float().flatten()
            h.update(torch.round(t * 1e4).to(torch.int64).numpy().tobytes())
        return h.hexdigest()
    
    def _bind_probe(self) -> None:
        """
        Attach the signal probe once the owning node is bound to the consensus strategy.
        Deferred binding ensures the node_id and run-specific output directory are fully initialized.
        """
        if self.signal_probe is None and self._probe_enabled and getattr(self, "node", None) is not None:
            resolved_dir = self._resolve_probe_output_dir()
            self.signal_probe = SignalProbe(
                node_id=self.node.node_id,
                enabled=True,
                output_dir=resolved_dir,
            )
            logger.info(f"[PROBE BIND] SignalProbe bound for node '{self.node.node_id}' at: {resolved_dir}")

    def _record_probe_row(
        self,
        proposer_id: str,
        reconstructed_weights: Dict[str, torch.Tensor],
        base_w: Dict[str, torch.Tensor],
        actual_tested_acc: float,
        parents_tested_acc: float,
        m3: float,
        claimed_classes: Dict[int, float],
        block_timestamp: float,
        parent_hashes: Optional[List[str]] = None,
    ) -> None:
        """
        Assemble and persist one probe observation for offline component-selection analysis.
        Implements defensive execution to ensure monitoring never disrupts the consensus flow.
        """
        # Guard Clause: Bypass immediately if probe is disabled or not initialized
        if not self._probe_enabled or self.signal_probe is None:
            return

        try:
            # 1. Fetch previous model from cache for temporal drift metrics (c2_cos, c6_self_jump)
            prev_weights = None
            prev_cid = self._proposer_last_cid.get(proposer_id)
            if prev_cid:
                prev_weights = self.local_hot_cache.get(f"{prev_cid}_full")

            simulator = getattr(self.node, "simulator", None) if hasattr(self, "node") and self.node else None
            sim_time = simulator.current_time if simulator else 0.0

            # 2. Structural DAG Staleness Calculation (Age relative to parents)
            staleness = 0.0
            if simulator and parent_hashes:
                parent_times = [
                    self.node.local_dag.vertices[ph]["timestamp"]
                    for ph in parent_hashes
                    if hasattr(self.node, "local_dag") and ph in self.node.local_dag.vertices
                ]
                if parent_times:
                    staleness = max(0.0, sim_time - (sum(parent_times) / len(parent_times)))
                else:
                    staleness = max(0.0, sim_time - block_timestamp)
            else:
                staleness = max(0.0, sim_time - block_timestamp)

            # 3. Extract stateless signals with microsecond timing
            row = self.signal_extractor.extract(
                weights=reconstructed_weights,
                base_weights=base_w if base_w else {},
                prev_weights=prev_weights,
                accuracy=actual_tested_acc,
                parent_accuracy=parents_tested_acc,
                m3=m3,
                per_class=claimed_classes,
                staleness=staleness,
            )

            # 4. Extract Ground-Truth label securely from the DES Oracle
            label = 0
            if simulator and hasattr(simulator, "nodes"):
                proposer_node = simulator.nodes.get(proposer_id)
                if proposer_node:
                    label = int(bool(getattr(proposer_node, "is_malicious", False)))

            # 5. Attach metadata and record synchronously
            row.update({
                "t": sim_time,
                "validator": self.node.node_id if self.node else "unknown",
                "proposer": proposer_id,
                "label": label,
            })
            self.signal_probe.record(row)

        except Exception as e:
            # Defensive logging: Prevent probe failures from propagating to consensus engine
            logger.error(f"[PROBE ERROR] Failed to record row for proposer '{proposer_id}': {e}", exc_info=True)

    def _resolve_probe_output_dir(self) -> str:
        """
        Safely resolves the per-run output directory for probe CSV files across varying runtime configurations.
        """
        sim_instance = getattr(self.node, "simulator", None) if hasattr(self, "node") and self.node else None
        
        candidates = [
            self.config.get("output", {}).get("directory") if isinstance(self.config.get("output"), dict) else None,
            self.config.get("output", {}).get("dir") if isinstance(self.config.get("output"), dict) else None,
            self.config.get("simulation", {}).get("output_dir") if isinstance(self.config.get("simulation"), dict) else None,
            self.config.get("output_root"),
            getattr(sim_instance, "output_dir", None),
            getattr(sim_instance, "results_dir", None),
        ]
        
        for path in candidates:
            if path and isinstance(path, str) and path.strip():
                os.makedirs(path, exist_ok=True)
                return path
                
        return "results"
    
    def _calculate_class_entropy(self, class_accuracies: Dict[int, float]) -> float:
        """
        Calculate Shannon entropy of class accuracy distribution.
        
        Args:
            class_accuracies: Dictionary of per-class accuracies.
            
        Returns:
            Shannon entropy (0 to log(num_classes)).
        """
        if not class_accuracies:
            return 0.0
        
        masses = [max(float(v), 0.0) for v in class_accuracies.values()]
        total = sum(masses)
        if total <= 0.0:
            return 0.0
        
        probs = [m / total for m in masses if m > 0.0]
        return -sum(p * math.log(p) for p in probs)