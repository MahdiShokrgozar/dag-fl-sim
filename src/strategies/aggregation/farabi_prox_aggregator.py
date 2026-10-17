"""
Module: farabi_prox_aggregator.py
Description:
    Implements the FARABI Advanced Proximal Aggregator (Farabi-Prox).
    Solves Catastrophic Forgetting in highly Non-IID environments by using
    Anchor-based Proximal Dampening alongside Reputation (Virtue) Weighting:

        Weight_i = Virtue_i * exp(-mu * ||W_i - W_anchor||_2) / sum(...)

    where the anchor is the parent model with the highest Virtue score.

    Restored from git history (commit 05e1395 introduced it; later replaced on
    the live default path by FarabiRobustAggregator) for a direct benchmark
    against that replacement. See thesis-hub/docs/decisions.md (2026-09-02) and
    federated-dag-fl/docs/farabi-protocol-spec.md for the context of why both
    exist and what this comparison is for.

    To keep the benchmark apples-to-apples (same weight-fetch/payload-parsing
    pipeline, only the aggregation math differs), this subclasses
    FarabiRobustAggregator and reuses its Genesis/metadata-block filtering and
    its `_lazy_fetch_weights` IPFS-fetch helper. Only `aggregate()` is
    overridden with the proximal-dampening math; `client_update()` is
    inherited unchanged (both aggregators train identically).
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class FarabiProxAggregator(FarabiRobustAggregator):
    agg_type = "farabi_prox"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.proximal_mu = (
            self.cfg.get("protocols", {})
            .get("farabi", {})
            .get("aggregation", {})
            .get("proximal_mu", 2.0)
        )
        # Staleness weight (thesis-hub/docs/decisions.md, 2026-09-25): same
        # config block and same age unit (rounds of the aggregating node, set on
        # the node by FarabiTipSelector) as the selection-side penalty, so an
        # old parent that still gets picked also counts for less.
        st_cfg = (
            self.cfg.get("protocols", {}).get("farabi", {})
            .get("selection", {}).get("staleness", {}) or {}
        )
        self.staleness_enabled = bool(st_cfg.get("enabled", False))
        self.staleness_lambda_aggregate = float(st_cfg.get("lambda_aggregate", 0.5))
        logger.info(f"⚖️ [FARABI-PROX] Initialized (mu={self.proximal_mu}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        # Same substitution as the parent class: in dual mode the set the core
        # handed over (tip parents, chosen for consensus speed) is not the set
        # this protocol should merge (finalized anchors, chosen for trust).
        # Inherited rather than duplicated because this class overrides
        # aggregate() wholesale.
        payloads = self._resolve_aggregation_inputs(payloads, node)
        # W4 (2026-09-26): what this call actually merged, as [(block_hash,
        # coefficient)], for FarabiConsensus.propose_block to declare. Cleared
        # first so a failed/aborted aggregation never declares a stale set.
        if node is not None:
            node.last_aggregation_coefficients = None

        if not payloads:
            logger.error("❌ [FARABI-PROX] Payloads list is empty. Aborting.")
            return {}

        # --- Stage 0: filter out empty/Genesis/metadata-only blocks (shared with FarabiRobustAggregator) ---
        valid_blocks = []
        for p in payloads:
            if not isinstance(p, dict):
                continue
            inner = p.get("payload", p)
            if isinstance(inner, dict) and len(inner.keys()) <= 2 and "metadata" in inner:
                continue
            block_hash = p.get("hash", "")
            is_genesis = (
                block_hash.startswith("00000000")
                or block_hash == ""
                or (inner.get("ipfs_cid") is None and "weights" not in inner and "update" not in inner)
            )
            if is_genesis:
                continue
            valid_blocks.append(p)

        if not valid_blocks:
            logger.error("❌ [FARABI-PROX] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights via the shared lazy-IPFS-fetch helper ---
        fetched: List[Dict[str, Any]] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if not weights:
                continue

            inner = payload.get("payload", payload)
            author = (
                payload.get("author")
                or payload.get("proposer")
                or inner.get("author")
                or inner.get("proposer")
                or "unknown"
            )
            fetched.append({"node_id": author, "weights": weights, "timestamp": payload.get("timestamp"),
                            "hash": payload.get("hash", "")})

        if not fetched:
            logger.error("❌ [FARABI-PROX] No valid models fetched. Aborting.")
            return {}

        # --- Stage 1b: weight-space anomaly filter (inherited from
        # FarabiRobustAggregator) - closes the gap that let a node adopt a
        # single poisoned tip verbatim as its aggregation result whenever
        # only one parent was available (N=1), with no trust-anchor check
        # against the node's own last-known-good weights. Confirmed root
        # cause of the 2026-09-14 loss-explosion cascade investigation (see
        # thesis-hub/docs/decisions.md, same date): node_6 adopted node_4's
        # (a poisoning attacker) single-tip block verbatim and exploded to
        # loss ~5.4e8 on the very first batch. This filter was already
        # implemented and used by the parent class's own aggregate(); this
        # class overrode aggregate() wholesale and never called it.
        if self._prefilter_enabled(arm_default=True):
            fetched = self._filter_weight_space_anomalies(fetched, node)
        if not fetched:
            logger.error("❌ [FARABI-PROX] All models dropped by anomaly filter. Aborting.")
            return {}

        models_to_aggregate: List[Dict[str, torch.Tensor]] = []
        reputation_scores: List[float] = []
        authors: List[str] = []
        hashes: List[str] = []
        staleness: List[float] = []
        period = getattr(node, "staleness_round_period", None) if node else None
        now = float(getattr(getattr(node, "simulator", None), "current_time", 0.0) or 0.0)
        for item in fetched:
            author = item["node_id"]
            virtue = 50.0
            if node and hasattr(node, "reputation"):
                virtue = node.reputation.get_virtue_score(author)
            virtue = max(virtue, 0.1)  # prevent zero division

            models_to_aggregate.append(item["weights"])
            reputation_scores.append(virtue)
            authors.append(author)
            hashes.append(item.get("hash", ""))
            factor = 1.0
            ts = item.get("timestamp")
            if self.staleness_enabled and period and ts is not None:
                age_rounds = max(0.0, now - float(ts)) / period
                factor = math.exp(-self.staleness_lambda_aggregate * age_rounds)
            staleness.append(factor)

        # --- Stage 2: anchor selection (highest-virtue model) ---
        anchor_idx = reputation_scores.index(max(reputation_scores))
        anchor_model = models_to_aggregate[anchor_idx]
        logger.info(
            f"⚓ [FARABI-PROX] Anchor: {authors[anchor_idx]} (virtue={reputation_scores[anchor_idx]:.2f})"
        )

        # --- Stage 3: L2 distance to anchor + proximal-dampened effective weight ---
        effective_weights = []
        for i, model in enumerate(models_to_aggregate):
            if i == anchor_idx:
                dist = 0.0
            else:
                dist_sq = sum(
                    (model[k] - anchor_model[k]).float().norm().item() ** 2
                    for k in model
                    if k in anchor_model
                )
                dist = math.sqrt(dist_sq)

            prox_factor = math.exp(-self.proximal_mu * dist)
            effective_weights.append(reputation_scores[i] * prox_factor * staleness[i])

        total_eff_weight = sum(effective_weights)
        if total_eff_weight <= 0:
            normalized_weights = [1.0 / len(effective_weights)] * len(effective_weights)
        else:
            normalized_weights = [w / total_eff_weight for w in effective_weights]

        # --- Stage 4: weighted fusion across all layers (no feature/classifier split) ---
        aggregated_weights: Dict[str, torch.Tensor] = {}
        all_keys = set()
        for m in models_to_aggregate:
            all_keys.update(m.keys())

        for key in all_keys:
            weighted_tensor_sum = None
            for i, model in enumerate(models_to_aggregate):
                if key in model:
                    tensor_scaled = model[key].float() * normalized_weights[i]
                    weighted_tensor_sum = (
                        tensor_scaled if weighted_tensor_sum is None else weighted_tensor_sum + tensor_scaled
                    )
            if weighted_tensor_sum is not None:
                aggregated_weights[key] = weighted_tensor_sum

        if node is not None and all(hashes):
            node.last_aggregation_coefficients = list(zip(hashes, normalized_weights))
        logger.info(f"✅ [FARABI-PROX] Aggregated {len(models_to_aggregate)} models.")
        return aggregated_weights
