"""
Module: farabi_selection.py
Description:
    Implements Dynamic Annealed Tip Selection (DATS) v4.0 with three-phase selection.
    
    Architecture (Three-Phase Selection):
    - Phase 1: Filter candidates (remove slashed/quarantined/low-virtue)
    - Phase 2: Virtue-weighted random sampling WITHOUT replacement (select n)
    - Phase 3: Metadata-based filtering (select k based on exploration/exploitation)
    
    Key Features:
    - Sub-linear scaling (sqrt) for Sybil resistance
    - Sampling WITHOUT replacement (prevents duplicate selection)
    - Dynamic n based on network size
    - Compatible with both DAG structures (blocks and vertices)
"""

import math
import time
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

from src.strategies.selection.base import TipSelector

logger = logging.getLogger(__name__)


class FarabiTipSelector(TipSelector):
    """
    Dynamic Adaptive Tip Selection with three-phase architecture.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        selection_cfg = self.config.get("protocols", {}).get("farabi", {}).get("selection", {})
        
        # Epsilon-greedy parameters
        self.eps_min = selection_cfg.get("eps_min", 0.1)
        self.eps_max = selection_cfg.get("eps_max", 0.9)
        self.decay_rate = selection_cfg.get("decay_rate", 5.0)
        
        # Selection parameters
        self.num_tips = selection_cfg.get("num_tips", 3)
        self.analytical_delay = selection_cfg.get("analytical_delay", 0.005)
        
        # Dynamic candidate pool parameters
        self.min_candidate_pool = selection_cfg.get("min_candidate_pool", 5)
        self.candidate_pool_ratio = selection_cfg.get("candidate_pool_ratio", 0.5)
        
        # Fairness and recency
        self.fairness_constant = selection_cfg.get("fairness_constant", 1.0)
        self.gamma_recency = selection_cfg.get("gamma_recency", 0.05)
        self.candidate_pool_size = selection_cfg.get("candidate_pool_size", 10)
        
        # Sybil resistance
        self.min_virtue_threshold = selection_cfg.get("min_virtue_threshold", 20.0)

        # Soft L2-distance trust signal (thesis-hub/docs/decisions.md,
        # 2026-09-03): replaces an earlier version of this fix that used a
        # hard self-reported-accuracy threshold and an invented "bootstrap"
        # phase counter - student correctly pointed out (1) we already
        # measure weight-space distance elsewhere, it should be a soft
        # signal here too, not a threshold, and (2) we already have a
        # phase engine (epsilon-greedy exploration/exploitation below), no
        # need for a second one. l2_distance_scale reuses the same 50.0
        # reference point _filter_weight_space_anomalies (farabi_aggregator.py)
        # already uses for its N==1 case, for consistency rather than a
        # fresh arbitrary number.
        # Staleness (thesis-hub/docs/decisions.md, 2026-09-25). The documented
        # DATS carries an exponential staleness penalty; the aggregation path
        # had one (score = recency * ...) until 8b1b202 (2026-08-20) dropped it
        # with no recorded reason, and the recency term left in the validation
        # path is dead (base_node never passes current_time, so age is always
        # 0). Plans 42-43 then showed a fixed-seconds tip-age cap only works
        # in a narrow window that does not transfer between workloads (3s
        # good on P13, 1.5s slashes honest nodes, 6s does nothing, no effect
        # on CIFAR). So age is measured in ROUNDS - seconds divided by the
        # node's own EMA round period - and penalised softly, both in the
        # chance of being selected here and in the aggregation weight
        # (FarabiProxAggregator). max_age_rounds is an optional hard backstop
        # that never leaves fewer than num_tips candidates. Off by default.
        st_cfg = selection_cfg.get("staleness", {}) or {}
        self.staleness_enabled = bool(st_cfg.get("enabled", False))
        self.staleness_lambda_select = float(st_cfg.get("lambda_select", 0.5))
        self.staleness_max_age_rounds = st_cfg.get("max_age_rounds", None)
        self.staleness_period_ema = float(st_cfg.get("period_ema", 0.3))
        self._last_select_time: Dict[str, float] = {}
        self._round_period: Dict[str, float] = {}

        self.l2_distance_scale = selection_cfg.get("l2_distance_scale", 50.0)
        # Where the weight-space distance is computed (thesis-hub/docs/decisions.md,
        # 2026-10-01, architecture paper): "all" (default, every run so far) computes
        # it for every tip in view, O(L*P) per selection with L growing with the pool;
        # "candidates" draws the candidate_pool_size candidates from the local view
        # without it (sqrt-virtue x age weights) and computes it only for them,
        # O(L + c*P), so the P-heavy part no longer grows with the pool.
        self.l2_distance_scope = selection_cfg.get("l2_distance_scope", "all")
        if self.l2_distance_scope not in ("all", "candidates"):
            raise ValueError(f"selection.l2_distance_scope must be 'all' or 'candidates', got {self.l2_distance_scope!r}")
        self.l2_trust_exponent_exploration = selection_cfg.get("l2_trust_exponent_exploration", 3.0)
        self.l2_trust_exponent_exploitation = selection_cfg.get("l2_trust_exponent_exploitation", 1.0)

        # ABLATION HANDLE for the Emergency Historical Fallback.
        #   "auto"   - current behaviour: fire only when the tip pool is empty
        #   "never"  - never fire; an empty pool anchors to Genesis instead
        #   "always" - take the historical path on EVERY selection, bypassing
        #              the normal DATS ranking entirely
        #
        # This exists because plan-22 stage 1 found the fallback count predicts
        # honest accuracy almost perfectly across 8 cells, three attack regimes
        # and two aggregators (r=0.985), while attack SEVERITY does not. That is
        # correlational: the fallback only fires when the pool starves, and the
        # pool only starves under attack, so the two cannot be separated by
        # observation. "always" and "never" break that coupling.
        # See thesis-hub/docs/decisions.md (2026-09-06).
        # DUAL-PATH SELECTION (thesis-hub/docs/decisions.md, 2026-09-06).
        #   "unified"  - shipped behaviour: ONE selection serves both the DAG
        #                parent set and the aggregation input.
        #   "dual"     - option A. Parents come from tips (consensus duty, fast);
        #                the aggregation base comes from FINALIZED blocks and
        #                travels in the block as `aggregation_refs`.
        #   "anchored" - option B. One list. `num_anchors` of the num_tips slots
        #                are filled from the finalized set instead of the tip
        #                pool, so the per-block fetch/verify budget is unchanged.
        #
        # The split exists because the two jobs have opposite requirements: the
        # parent set must be chosen fast (it is the scalability bottleneck) while
        # the aggregation input must be chosen for trust (it is the poisoning
        # surface). select_tips_for_validation and select_tips_for_aggregation
        # were both already written for this; nothing ever passed `purpose`, so
        # everything ran through the aggregation branch.
        self.selection_mode = str(selection_cfg.get("mode", "unified")).lower()
        if self.selection_mode not in ("unified", "dual", "anchored"):
            logger.warning(
                f"⚠️ [DATS] Unknown selection.mode '{self.selection_mode}'; using 'unified'."
            )
            self.selection_mode = "unified"
        self.num_anchors = int(selection_cfg.get("num_anchors", 1))
        self.anchor_pool_size = int(selection_cfg.get("anchor_pool_size", 10))

        self.emergency_fallback = str(selection_cfg.get("emergency_fallback", "auto")).lower()
        if self.emergency_fallback not in ("auto", "never", "always"):
            logger.warning(
                f"⚠️ [DATS] Unknown selection.emergency_fallback "
                f"'{self.emergency_fallback}'; falling back to 'auto'."
            )
            self.emergency_fallback = "auto"

        logger.info(
            f"🎯 FARABI Tip Selector (DATS v4.0) Initialized | "
            f"Eps: [{self.eps_min}-{self.eps_max}] | "
            f"Num Tips: {self.num_tips} | "
            f"Min Virtue: {self.min_virtue_threshold}"
        )

    def select_tips(
        self,
        dag: Any,
        node: Any,
        current_time: float = 0.0,
        purpose: str = "aggregation",
    ) -> Tuple[List[str], float]:
        """
        Main entry point - routes to purpose-specific implementation.

        The simulator core calls this with the generic TipSelector contract
        ("give me parent hashes") and knows nothing about FARABI's modes. Every
        mode below is resolved here, inside the protocol's own strategy, so the
        core orchestration stays protocol-agnostic.
        """
        if purpose == "validation":
            return self.select_tips_for_validation(dag, node, current_time)

        if self.selection_mode == "anchored" and purpose == "aggregation":
            # OPTION B, and note it needs nothing outside this method: the parent
            # list simply stops being all-tips. num_anchors of the num_tips slots
            # are filled from the FINALIZED set, so the block still declares one
            # list, the core still fetches num_tips models, and the per-block
            # verify budget is unchanged - but part of the aggregation base is
            # now drawn from blocks the network has already confirmed rather
            # than from unverified tips.
            anchors = self.select_anchors(dag, node, self.anchor_count())
            anchor_ids = [self._extract_hash(a) for a in anchors]
            tip_ids, delay = self.select_tips_for_validation(dag, node, current_time)
            room = max(0, self.num_tips - len(anchor_ids))
            return list(tip_ids[:room]) + anchor_ids, delay

        if self.selection_mode == "dual" and purpose == "aggregation":
            # OPTION A. The core is asking for PARENTS, and in dual mode those
            # are purely a consensus duty: take the fast validation path. The
            # aggregation base is resolved separately, by the aggregator, from
            # the finalized set - see FarabiRobustAggregator._resolve_aggregation_inputs.
            return self.select_tips_for_validation(dag, node, current_time)

        if purpose == "aggregation":
            return self.select_tips_for_aggregation(dag, node, current_time)

        logger.error(f"Unknown purpose: {purpose}. Defaulting to aggregation.")
        return self.select_tips_for_aggregation(dag, node, current_time)

    def _get_all_tips_from_dag(self, dag: Any) -> List[Dict[str, Any]]:
        """
        Extract all tips from DAG, compatible with both FarabiDAGManager and legacy DAG.
        
        Supports:
        - dag.get_available_tips() (FarabiDAGManager)
        - dag.blocks (FarabiDAGManager)
        - dag.vertices (Legacy DAG)
        - dag.get_tips() (Legacy interface)
        """
        tips = []
        
        # Method 1: get_available_tips() - preferred for FarabiDAGManager
        if hasattr(dag, 'get_available_tips'):
            try:
                result = dag.get_available_tips()
                if result:
                    return result
            except Exception as e:
                logger.debug(f"get_available_tips() failed: {e}")
        
        # Method 2: dag.blocks dict (FarabiDAGManager)
        if hasattr(dag, 'blocks') and isinstance(dag.blocks, dict):
            try:
                if hasattr(dag, 'tips') and isinstance(dag.tips, set):
                    # sorted(): dag.tips is a set - unsorted iteration order
                    # depends on PYTHONHASHSEED, making tip selection (and
                    # thus which models get aggregated) non-reproducible.
                    for tip_id in sorted(dag.tips):
                        if tip_id in dag.blocks:
                            tips.append(dag.blocks[tip_id])
                    if tips:
                        return tips
            except Exception as e:
                logger.debug(f"dag.blocks access failed: {e}")
        
        # Method 3: dag.vertices dict (Legacy DAG)
        if hasattr(dag, 'vertices') and isinstance(dag.vertices, dict):
            try:
                if hasattr(dag, 'tips') and isinstance(dag.tips, set):
                    # sorted(): see determinism note in Method 2 above.
                    for tip_id in sorted(dag.tips):
                        if tip_id in dag.vertices:
                            tips.append(dag.vertices[tip_id])
                    if tips:
                        return tips
            except Exception as e:
                logger.debug(f"dag.vertices access failed: {e}")
        
        # Method 4: dag.get_tips() (Legacy interface)
        if hasattr(dag, 'get_tips'):
            try:
                raw_tips = dag.get_tips()
                for t in raw_tips:
                    if isinstance(t, dict):
                        tips.append(t)
                    elif hasattr(dag, 'get_block'):
                        blk = dag.get_block(t)
                        if blk:
                            tips.append(blk)
                if tips:
                    return tips
            except Exception as e:
                logger.debug(f"dag.get_tips() failed: {e}")
        
        return tips

    def _get_filtered_tips(self, dag: Any, node: Any) -> List[Dict[str, Any]]:
        """
        Get tips with filtering of malicious/quarantined/slashed nodes.
        Uses L2 strategy's local view when available for efficiency.
        """
        available_tips = []
        node_id = node.node_id if node else "unknown"
        current_time = getattr(node.simulator, 'current_time', 0.0) if node and hasattr(node, 'simulator') else 0.0
        
        # Try L2 strategy's local view first (most efficient)
        if node and hasattr(node, 'l2_strategy') and hasattr(node.l2_strategy, 'get_local_view'):
            try:
                visible_tips, visible_history = node.l2_strategy.get_local_view(node_id, current_time)
                if visible_tips:
                    available_tips = list(visible_tips)
            except Exception as e:
                logger.debug(f"get_local_view failed: {e}")
        
        # Fallback to direct DAG access
        if not available_tips:
            available_tips = self._get_all_tips_from_dag(dag)
        
        if not available_tips:
            return []
        
        # Filter
        valid_tips = []
        reputation_manager = node.reputation if node and hasattr(node, 'reputation') else None
        locally_slashed = set()
        if node and hasattr(node, 'l2_strategy') and hasattr(node.l2_strategy, 'locally_slashed_nodes'):
            locally_slashed = node.l2_strategy.locally_slashed_nodes
        
        # Was `hasattr(dag, 'finalized_blocks')`, which is always False - finality
        # is L2 state on node.l2_strategy, not on the DAG object selection is
        # handed. The staleness filter below has therefore never fired.
        finalized_blocks, _ = self._finality_view(dag, node)

        for tip in available_tips:
            if not isinstance(tip, dict):
                continue
            
            block_id = tip.get('hash', tip.get('block_id', tip.get('id', '')))
            proposer = tip.get("proposer", tip.get("author", "unknown"))
            
            # Skip finalized
            if block_id in finalized_blocks:
                continue
            
            # Skip quarantined
            if reputation_manager and reputation_manager.is_quarantined(proposer):
                continue
            
            # Skip slashed
            if proposer in locally_slashed:
                continue
            
            # Skip low-virtue nodes (Sybil resistance)
            if reputation_manager:
                virtue = reputation_manager.get_virtue_score(proposer)
                if virtue < self.min_virtue_threshold:
                    continue
            
            valid_tips.append(tip)
        
        logger.info(f"[SELECTION-DEBUG] Node {node.node_id}: "
                f"Total tips: {len(available_tips)}, "
                f"After filter: {len(valid_tips)}")
        
        return valid_tips

    def select_tips_for_validation(
        self,
        dag: Any,
        node: Any,
        current_time: float,
    ) -> Tuple[List[str], float]:
        """
        Validation: Select tips with high class similarity.
        No exploration - only exploitation of known similar distributions.
        """
        available_tips = self._get_filtered_tips(dag, node)
        
        if not available_tips:
            logger.warning(f"[VALIDATION] No valid tips available for {node.node_id}")
            return [], self.analytical_delay
        
        # Calculate class similarity for each tip
        scored_tips = []
        local_classes = getattr(node, 'per_class_accuracy', {})
        
        for tip in available_tips:
            tip_classes = tip.get('class_accuracies', tip.get('payload', {}).get('class_accuracies', {}))
            similarity = self._calculate_class_similarity(local_classes, tip_classes)
            
            proposer = tip.get("proposer", tip.get("author", ""))
            virtue = node.reputation.get_virtue_score(proposer) if node and hasattr(node, 'reputation') else 50.0
            
            # Sub-linear scaling for Sybil resistance
            trust_power = math.sqrt(max(virtue, 0.0) / 100.0)
            
            # Recency bonus
            tip_time = float(tip.get('timestamp', current_time))
            time_diff = max(0.0, float(current_time) - tip_time)
            recency = math.exp(-self.gamma_recency * time_diff)
            
            # Final score: similarity-weighted trust
            final_score = similarity * trust_power * (1.0 + recency)
            scored_tips.append((tip, final_score, similarity))
        
        # Sort and select top K
        scored_tips.sort(key=lambda x: x[1], reverse=True)
        selected = [tip for tip, score, sim in scored_tips[:self.num_tips]]
        
        selected_ids = [t.get('hash', t.get('block_id', 'unknown_hash')) for t in selected]
        
        logger.info(
            f"✅ [VALIDATION] Node {node.node_id} selected {len(selected_ids)} tips"
        )
        
        return selected_ids, self.analytical_delay

    def _apply_security_filters(self, blocks: List[Dict[str, Any]], node: Any, dag: Any, skip_finalized: bool = True) -> List[Dict[str, Any]]:
        """
        Centralized security filter to sanitize the candidate pool.
        Removes blocks proposed by malicious, quarantined, or low-virtue nodes.
        
        Args:
            blocks: List of block dictionaries to be filtered.
            node: Reference to the local node executing the selection.
            dag: Reference to the local or global DAG view.
            skip_finalized: If True, finalized blocks are removed (used for active tips).
                            If False, finalized blocks are kept (used for historical fallback).
                            
        Returns:
            List[Dict[str, Any]]: A sanitized list of candidate blocks.
        """
        valid_blocks = []
        reputation_manager = node.reputation if node and hasattr(node, 'reputation') else None
        
        # Retrieve the set of locally slashed nodes (Layer-2 isolation)
        locally_slashed = set()
        if node and hasattr(node, 'l2_strategy') and hasattr(node.l2_strategy, 'locally_slashed_nodes'):
            locally_slashed = node.l2_strategy.locally_slashed_nodes
            
        # Retrieve the set of finalized blocks to prevent aggregating stale or
        # already-settled features. Same latent bug as above: this guarded on
        # `hasattr(dag, 'finalized_blocks')`, which never holds.
        finalized_blocks, _ = self._finality_view(dag, node)

        my_id = node.node_id if node else "unknown"

        for b in blocks:
            if not isinstance(b, dict): 
                continue
            
            block_id = b.get('hash', b.get('block_id', b.get('id', '')))
            proposer = b.get("proposer", b.get("author", "unknown"))

            # 1. Anti-Self-Looping: Prevent a node from validating/aggregating its own immediate block
            if proposer == my_id:
                continue

            # 2. Staleness Filter: Ignore finalized blocks during the primary active tip selection
            if skip_finalized and block_id in finalized_blocks:
                continue
                
            # 3. Byzantine Isolation: Drop blocks from actively quarantined or slashed nodes
            if reputation_manager and reputation_manager.is_quarantined(proposer):
                continue
            if proposer in locally_slashed:
                continue
                
            # 4. Sybil Resistance: Enforce a minimum virtue threshold for participation
            if reputation_manager:
                virtue = reputation_manager.get_virtue_score(proposer)
                if virtue < self.min_virtue_threshold:
                    continue
                    
            valid_blocks.append(b)
            
        return valid_blocks


    def _get_historical_blocks(self, dag: Any) -> List[Dict[str, Any]]:
        """
        Extracts the full block history directly from the DAG data structures.
        Bypasses missing API methods by safely inspecting standard properties.
        """
        if hasattr(dag, 'blocks') and isinstance(dag.blocks, dict):
            return list(dag.blocks.values())
        if hasattr(dag, 'vertices') and isinstance(dag.vertices, dict):
            return list(dag.vertices.values())
        return []

    def anchor_count(self, for_aggregation: bool = False) -> int:
        """
        How many finalized anchors this mode should pull.

        The two modes want different numbers and neither is `num_anchors` alone,
        so the policy lives here rather than being restated at each call site.

        dual      the anchors REPLACE the parent set as the aggregation input,
                  so there must be as many of them as `unified` would have
                  aggregated - i.e. num_tips. The plan-23 crash check ran with
                  num_anchors=1 and aggregated exactly one model in 237 of 241
                  rounds, which is "copy one finalized block", not averaging.
                  Plan 22 already established that single-model aggregation
                  tracks poor accuracy.
        anchored  the anchors CONSUME parent slots, so at most num_tips - 1:
                  approving tips is the consensus duty, and a node that spent
                  every slot on already-finalized blocks would approve no tips
                  at all and stop contributing to DAG growth.
        """
        if for_aggregation and self.selection_mode == "dual":
            return self.num_tips
        return max(1, min(self.num_anchors, self.num_tips - 1))

    @staticmethod
    def _finality_view(dag: Any, node: Any) -> Tuple[set, Dict[str, Any]]:
        """
        Locate the finalized-block set and the block store that holds them.

        Finality is L2 state: `finalized_blocks` and `blocks` live on
        FarabiDAGManager, i.e. on `node.l2_strategy`. The object handed to
        selection is `active_dag`, a plain `src.dag.dag.DAG`, which has no such
        attribute. Two existing call sites guard with
        `hasattr(dag, 'finalized_blocks')`, which is therefore always False, so
        their `# Skip finalized` filters have never once fired - and the plan-23
        crash check found the same thing in select_anchors: 55 blocks finalized,
        zero anchors located, both new modes silently degrading to `unified`.

        Checks the DAG first so a future DAG that does carry finality still
        works, then falls back to the L2 strategy.
        """
        finalized = getattr(dag, 'finalized_blocks', None)
        store = None
        if finalized:
            store = getattr(dag, 'blocks', None) or getattr(dag, 'vertices', None)

        if not finalized:
            l2 = getattr(node, 'l2_strategy', None) if node else None
            finalized = getattr(l2, 'finalized_blocks', None)
            if finalized:
                store = getattr(l2, 'blocks', None)

        if not finalized or not store:
            return set(), {}
        return set(finalized), store

    def select_anchors(self, dag: Any, node: Any, k: int) -> List[Dict[str, Any]]:
        """
        Pick `k` blocks from the FINALIZED set to serve as the aggregation base.

        A finalized block has already cleared the network's confirmation
        threshold (`weight >= max(w_min, theta * mean_virtue)`), so its model is
        trusted in a way a raw tip is not. That is the whole point of sourcing
        the aggregation base here instead of from the tip pool: a poisoned model
        can sit in the tip pool, but it cannot be finalized without the network
        having weighted it.

        The explore/exploit balance lives HERE rather than in tip selection.
        Both halves of it are wanted for aggregation - unlike models widen class
        coverage under Non-IID, similar models converge - and applying it to a
        pre-trusted pool is what it was designed for. Tip selection, which only
        has to keep the DAG growing quickly, does not need it.

        Returns [] when nothing is finalized yet; the caller handles cold start.
        """
        if k <= 0:
            return []

        finalized_ids, blocks_by_id = self._finality_view(dag, node)
        if not finalized_ids:
            return []

        candidates = [blocks_by_id[bid] for bid in sorted(finalized_ids) if bid in blocks_by_id]
        if not candidates:
            return []

        # Same security filters as any other candidate pool, minus the staleness
        # rule - "finalized" is precisely what we are selecting for here.
        candidates = self._apply_security_filters(candidates, node, dag, skip_finalized=False)
        if not candidates:
            return []

        # Restrict to the most recent slice before ranking. Ranking the whole
        # finalized history would drift toward stale models, and everyone
        # anchoring on the single newest block would concentrate the DAG on one
        # vertex - anchor_pool_size is the knob between those two failures.
        candidates.sort(key=lambda b: b.get('timestamp', 0.0), reverse=True)
        pool = candidates[:max(k, self.anchor_pool_size)]

        local_accuracy = getattr(node, 'node_accuracy', 0.0) if node else 0.0
        norm_acc = min(max(local_accuracy, 0.0), 100.0) / 100.0
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.decay_rate * norm_acc)
        is_exploration = random.random() < epsilon
        exponent = (
            self.l2_trust_exponent_exploration if is_exploration
            else self.l2_trust_exponent_exploitation
        )

        local_classes = getattr(node, 'per_class_accuracy', {})
        scored = []
        for b in pool:
            tip_classes = b.get('class_accuracies', b.get('payload', {}).get('class_accuracies', {}))
            similarity = self._calculate_class_similarity(local_classes, tip_classes)
            score = (1.0 - similarity) if is_exploration else similarity
            score *= self._l2_trust_multiplier(b, node, exponent)
            scored.append((b, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = [b for b, _ in scored[:k]]
        logger.info(
            f"⚓ [ANCHOR] Node {getattr(node,'node_id','?')} selected {len(chosen)} "
            f"finalized anchor(s) from a pool of {len(pool)} "
            f"({'exploration' if is_exploration else 'exploitation'})."
        )
        return chosen

    def _emergency_historical_tips(
        self, dag: Any, rep_manager: Any, node_id: str
    ) -> List[Dict[str, Any]]:
        """
        The Emergency Historical Fallback tip set: the `num_tips` most recent
        blocks in the whole DAG, quarantined proposers excluded.

        Extracted from the inline fallback so the ablation arms
        (selection.emergency_fallback) all exercise the same code, and so the
        difference between this path and the normal DATS ranking is legible in
        one place. That difference is the point: this path bypasses the
        self-loop filter, min_virtue_threshold, the L2 trust multiplier AND the
        Phase-3 cosine alignment. Under exploitation Phase 3 scores tips by
        SIMILARITY to the local model, which on Dirichlet alpha=0.1 means
        preferring peers holding the same classes - so the normal path can
        stall class coverage while this one, being a plain recency slice across
        all proposers, keeps injecting unlike models.

        Returns [] when the DAG has no usable history; the caller decides what
        to do about that.
        """
        history = self._get_historical_blocks(dag)

        emergency_tips = []
        for b in history:
            proposer = self._extract_proposer(b)
            # The self-loop filter is deliberately relaxed here - a node may
            # build on its own previous block - because the whole point is to
            # avoid deadlocking when nothing else is available. Quarantine is
            # still enforced: security is not relaxed, only topology.
            if rep_manager and rep_manager.is_quarantined(proposer):
                continue
            emergency_tips.append(b)

        if not emergency_tips:
            return []

        emergency_tips.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        selected = emergency_tips[:self.num_tips]

        # Logged HERE rather than at the call sites so the count is comparable
        # across modes. Stage 1b had the "Engaging Emergency Historical Fallback"
        # warning only on the auto branch, so `always` runs reported 0 fallbacks
        # while actually taking this path on every single selection - the one
        # number the whole ablation turns on read as its own opposite.
        logger.info(
            f"📚 [DATS] Historical tip path used (mode={self.emergency_fallback}). "
            f"Selected {len(selected)} tips."
        )
        return selected

    @staticmethod
    def _genesis_anchor() -> List[Dict[str, Any]]:
        """Last resort when the DAG holds nothing usable (e.g. epoch 0)."""
        logger.info("❄️ [DATS] Absolute cold start. Anchoring to Genesis.")
        return [{'hash': "0" * 64, 'author': 'system', 'timestamp': 0.0}]

    def _extract_proposer(self, block: Dict[str, Any]) -> str:
        """
        Safely extracts the proposer's identity from a block.
        To prevent 'Metadata Blindness', it rigorously inspects both the root 
        dictionary and the nested 'payload' dictionary.
        """
        payload = block.get('payload', {})
        return block.get('author') or block.get('proposer') or payload.get('author') or payload.get('proposer') or "unknown"
        
    def _extract_hash(self, block: Dict[str, Any]) -> str:
        """
        Safely extracts the block's unique cryptographic hash or ID.
        Traverses common key naming conventions at both root and nested levels.
        """
        payload = block.get('payload', {})
        return block.get('hash') or block.get('block_id') or block.get('id') or payload.get('hash') or "unknown"
    
    def _update_round_period(self, node: Any) -> None:
        """EMA of the time between this node's successive aggregation selections.

        Called once per selection, i.e. once per round. Published on the node
        so FarabiProxAggregator can express parent ages in the same unit.
        """
        if not node or not hasattr(node, 'simulator'):
            return
        now = float(getattr(node.simulator, 'current_time', 0.0))
        node_id = node.node_id
        last = self._last_select_time.get(node_id)
        self._last_select_time[node_id] = now
        if last is None or now <= last:
            return
        delta = now - last
        prev = self._round_period.get(node_id)
        a = self.staleness_period_ema
        period = delta if prev is None else a * delta + (1.0 - a) * prev
        self._round_period[node_id] = period
        node.staleness_round_period = period

    def age_in_rounds(self, block: Dict[str, Any], node: Any) -> float:
        """Block age in units of this node's round period; 0 until one is known.

        `timestamp` on an L2 block is its block_ready_time, the moment its
        model exists, so age counts from when the model became usable.
        """
        period = self._round_period.get(node.node_id) if node else None
        if not period or not hasattr(node, 'simulator'):
            return 0.0
        now = float(getattr(node.simulator, 'current_time', 0.0))
        ts = float(block.get('timestamp', now) or now)
        return max(0.0, now - ts) / period

    def staleness_factor(self, block: Dict[str, Any], node: Any) -> float:
        return math.exp(-self.staleness_lambda_select * self.age_in_rounds(block, node))

    def _apply_age_backstop(self, tips: List[Dict[str, Any]], node: Any) -> List[Dict[str, Any]]:
        """Drop tips older than max_age_rounds, but never below num_tips of them.

        Plan 43b: a 1.5s cap left 25-29% of aggregations single-model and got
        three honest nodes slashed; keeping at least num_tips (the youngest)
        avoids that failure mode by construction.
        """
        if self.staleness_max_age_rounds is None or len(tips) <= self.num_tips:
            return tips
        limit = float(self.staleness_max_age_rounds)
        aged = sorted(((self.age_in_rounds(t, node), t) for t in tips), key=lambda x: x[0])
        fresh = [t for a, t in aged if a <= limit]
        if len(fresh) >= self.num_tips:
            return fresh
        return [t for _, t in aged[:self.num_tips]]

    def select_tips_for_aggregation(
        self,
        dag: Any,
        node: Any,
        current_time: float,
    ) -> Tuple[List[str], float]:
        """
        Pure DAG Tip Selection Algorithm (DATS v4.0) - Enhanced Version.
        
        This method implements a three-phase selection strategy:
        1. Phase 1: Security Filtering (Quarantine, Slash, Sybil Resistance)
        2. Phase 2: Topology-Aware Dynamic Selection with Emergency Fallback
        3. Phase 3: Metadata Alignment (Cosine Similarity for Non-IID)
        
        Key Enhancement:
        The "Genesis Trap" vulnerability has been fixed by implementing 
        an Emergency Historical Fallback. When the active tip pool is 
        empty (due to network latency or aggressive filtering), the 
        algorithm now falls back to recent historical blocks instead of 
        anchoring to the Genesis block. This prevents the DAG from 
        degenerating into a star topology centered on Genesis.
        
        Args:
            dag (Any): Reference to the local or global DAG.
            node (Any): Reference to the local node.
            current_time (float): Current simulation time.
        
        Returns:
            Tuple[List[str], float]: A tuple containing selected parent 
            hashes and the analytical delay for telemetry.
        """
        node_id = node.node_id if node else "unknown"
        rep_manager = getattr(node, 'reputation', None)
        _t_sel = time.perf_counter()
        if self.staleness_enabled:
            self._update_round_period(node)

        # Exploration/exploitation phase - computed once, up front, and
        # reused everywhere a "how much do we trust virtue yet" decision is
        # needed (thesis-hub/docs/decisions.md, 2026-09-03: reuse the
        # existing epsilon-greedy engine instead of inventing a second,
        # parallel phase concept). Early/uncertain (low local accuracy) ->
        # high epsilon -> exploration more likely -> lean harder on the
        # local L2-distance signal below, since virtue hasn't had time to
        # diverge between honest and malicious proposers yet.
        local_accuracy = getattr(node, 'node_accuracy', 0.0) if node else 0.0
        norm_acc = min(max(local_accuracy, 0.0), 100.0) / 100.0
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.decay_rate * norm_acc)
        is_exploration = random.random() < epsilon
        if node:
            node.current_selection_mode = "exploration" if is_exploration else "exploitation"
        l2_trust_exponent = (
            self.l2_trust_exponent_exploration if is_exploration
            else self.l2_trust_exponent_exploitation
        )

        # Fetch universally or locally visible active tips (excluding finalized blocks)
        candidates = self._get_filtered_tips(dag, node)

        # --- Phase 1: Deep Metadata Extraction & Sybil Resistance Filtering ---
        valid_tips = []

        for tip in candidates:
            tip_id = self._extract_hash(tip)
            proposer = self._extract_proposer(tip)

            # 1. Prevent self-looping and discard malformed metadata
            if proposer == node_id or tip_id == "unknown":
                continue

            # 2. Sybil & Byzantine Isolation Check
            if rep_manager:
                if rep_manager.is_quarantined(proposer):
                    continue
                if rep_manager.get_virtue_score(proposer) < self.min_virtue_threshold:
                    continue

            # Soft weight-space trust signal. NOTE (plan 77, 2026-10-01): under
            # the FARABI processor a published block carries only the IPFS CID
            # of its model (process_outbound offloads the weights), so tip_weights
            # is absent and this factor is always 1.0 - selection reads metadata
            # only, O(L), and the weight-space distance is paid in the merge, for
            # the k fetched parents. Kept inert by decision rather than activated,
            # since activating it would mean fetching every candidate's model.
            # It acts only for blocks that do carry weights inline. Continuous
            # discount (never a hard 0/1 cut), raised to a phase-dependent
            # power: punishing during exploration (no other reliable signal
            # yet), gentle during exploitation (virtue has had time to help).
            if self.l2_distance_scope == "all":
                tip['_l2_trust_multiplier'] = self._l2_trust_multiplier(tip, node, l2_trust_exponent)
            else:
                # tip dicts are shared across selections: drop a value left by an
                # earlier selection so the candidate draw sees no distance at all
                tip.pop('_l2_trust_multiplier', None)

            valid_tips.append(tip)

        selected_k = []

        # --- Phase 2: Topology-Aware Dynamic Selection (WITH EMERGENCY FALLBACK) ---
        if self.emergency_fallback == "always":
            # ABLATION ARM. Take the historical path unconditionally, so the
            # fallback fires on every selection instead of only when the pool
            # starves. If honest accuracy then rises toward the attacked runs,
            # the stage-1 correlation is causal and this path is the fix, not
            # the safety net.
            forced = self._emergency_historical_tips(dag, rep_manager, node_id)
            if forced:
                valid_tips = forced
            elif not valid_tips:
                valid_tips = self._genesis_anchor()
        elif not valid_tips:
            if self.emergency_fallback == "never":
                # ABLATION ARM. Reproduce the Genesis Trap the fallback was
                # written to prevent, to size what it is actually worth.
                logger.warning(
                    f"⚠️ [DATS] Tip pool empty for Node {node_id} and "
                    f"emergency_fallback='never'. Anchoring to Genesis."
                )
                valid_tips = self._genesis_anchor()
            else:
                logger.warning(
                    f"⚠️ [DATS] Tip pool empty for Node {node_id}. "
                    f"Engaging Emergency Historical Fallback to prevent Genesis Trap."
                )
                valid_tips = self._emergency_historical_tips(dag, rep_manager, node_id)
                if not valid_tips:
                    valid_tips = self._genesis_anchor()

        if self.staleness_enabled:
            valid_tips = self._apply_age_backstop(valid_tips, node)
            for t in valid_tips:
                t['_staleness_factor'] = self.staleness_factor(t, node)

        # --- Phase 2 (continued): Selection Logic ---
        if len(valid_tips) <= self.num_tips:
            # Sparse network condition: Accept all available tips
            selected_k = [self._extract_hash(t) for t in valid_tips]
        else:
            # Dense network condition: Execute Virtue-Weighted Sampling
            n = min(getattr(self, 'candidate_pool_size', 10), len(valid_tips))
            
            # Calculate selection probabilities with sub-linear scaling
            # for Sybil resistance and fairness, multiplied by each tip's
            # soft L2-distance trust factor computed in Phase 1.
            if rep_manager:
                probs = [
                    math.sqrt(max(rep_manager.get_virtue_score(self._extract_proposer(t)), 0.0) / 100.0)
                    * t.get('_l2_trust_multiplier', 1.0)
                    for t in valid_tips
                ]
            else:
                probs = [t.get('_l2_trust_multiplier', 1.0) for t in valid_tips]
            probs = [p * t.get('_staleness_factor', 1.0) for p, t in zip(probs, valid_tips)]

            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs] if sum_probs > 0 else [1.0 / len(valid_tips)] * len(valid_tips)

            # Weighted random sampling WITHOUT replacement
            try:
                selected_indices = np.random.choice(len(valid_tips), size=n, replace=False, p=probs)
            except ValueError:
                selected_indices = random.sample(range(len(valid_tips)), n)

            sampled_tips = [valid_tips[i] for i in selected_indices]
            if self.l2_distance_scope == "candidates":
                for tip in sampled_tips:
                    tip['_l2_trust_multiplier'] = self._l2_trust_multiplier(tip, node, l2_trust_exponent)

            # --- Phase 3: Metadata Alignment (Cosine Similarity) ---
            scored = []
            local_classes = getattr(node, 'per_class_accuracy', {})

            for tip in sampled_tips:
                tip_classes = tip.get('class_accuracies', tip.get('payload', {}).get('class_accuracies', {}))
                similarity = self._calculate_class_similarity(local_classes, tip_classes)

                # In exploration mode, invert similarity to seek diverse
                # feature representations. In exploitation mode, prefer
                # similar models for stable convergence.
                score = 1.0 - similarity if is_exploration else similarity
                score *= tip.get('_l2_trust_multiplier', 1.0)
                score *= tip.get('_staleness_factor', 1.0)
                scored.append((tip, score))
            
            # Extract the top 'k' hashes
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_k = [self._extract_hash(t) for t, _ in scored[:self.num_tips]]

        # --- Hardware Telemetry Simulation ---
        actual_delay = self.analytical_delay
        if node and hasattr(node, 'telemetry'):
            effective_cap = getattr(node, 'capacity', 1.0) / max(0.01, getattr(node, 'delay_multiplier', 1.0))
            actual_delay /= max(0.01, effective_cap)
            node.telemetry.register_compute(actual_delay)

        logger.info(f"[SELECTION-COST] node={node_id} tips={len(candidates)} scope={self.l2_distance_scope} "
                    f"ms={(time.perf_counter() - _t_sel) * 1000.0:.3f}")
        return selected_k, actual_delay
    
    def _l2_trust_multiplier(self, tip: Dict[str, Any], node: Any, exponent: float) -> float:
        """
        Soft, continuous trust discount based on L2 distance between a
        candidate tip's weights and this node's own current local weights -
        the same "distance to a locally-trusted anchor" idea already used
        in farabi_aggregator.py's _filter_weight_space_anomalies, computed
        here at selection time instead of only after fetch/aggregation.
        Works on both torch tensors and numpy arrays without importing
        torch directly - `(diff ** 2).sum()` is defined on either.

        Returns 1.0 (no discount) if either side's weights aren't
        available - we abstain, not penalize, when there's no evidence
        either way (e.g. this node's own local model hasn't trained yet).
        """
        local_weights = getattr(node, 'weights', None) if node else None
        tip_weights = tip.get('weights') or tip.get('payload', {}).get('weights')
        if not local_weights or not tip_weights:
            return 1.0

        try:
            dist_sq = 0.0
            for name, local_tensor in local_weights.items():
                if name in tip_weights:
                    diff = tip_weights[name] - local_tensor
                    dist_sq += float((diff ** 2).sum())
            distance = dist_sq ** 0.5
        except Exception as e:
            logger.debug(f"🕵️ [SELECTION] L2 distance computation failed, abstaining: {e}")
            return 1.0

        distance_factor = 1.0 / (1.0 + distance / self.l2_distance_scale)
        return distance_factor ** exponent

    def _calculate_class_similarity(self, dist1: Dict[int, float], dist2: Dict[int, float]) -> float:
        """
        Calculates Cosine Similarity for Class Accuracy distributions.
        """
        if not dist1 and not dist2:
            return 1.0
        if not dist1 or not dist2:
            return 0.0
        
        keys = set(dist1.keys()).union(set(dist2.keys()))
        
        dot_product = sum(dist1.get(k, 0.0) * dist2.get(k, 0.0) for k in keys)
        norm1 = math.sqrt(sum(v**2 for v in dist1.values()))
        norm2 = math.sqrt(sum(v**2 for v in dist2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)