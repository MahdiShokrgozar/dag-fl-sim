"""
Module: rfa_aggregator.py
Description:
    RFA (Pillutla, Kakade & Harchaoui, 2019/IEEE TSP 2022 -- "Robust
    Aggregation for Federated Learning") as a literature-faithful external
    baseline for the Farabi-Prox Q1 benchmark.

    Algorithm: instead of the coordinate-wise mean (FedAvg) or coordinate-wise
    median/trimmed-mean (FarabiRobustAggregator's own feature-layer path),
    RFA computes the (sample-weighted) GEOMETRIC MEDIAN of the full model
    vectors via Weiszfeld's algorithm -- a single robust point in the whole
    parameter space that minimizes the sum of weighted L2 distances to all
    submitted models, rather than treating each coordinate independently.
    This is a meaningfully different notion of "robust" than median/
    trimmed-mean/Krum, so it is a useful third comparison point, not a
    near-duplicate of what FarabiRobustAggregator already does internally.

    Same fetch/filter pipeline reuse rationale as multi_krum_aggregator.py
    and bulyan_aggregator.py: reuses _resolve_aggregation_inputs and
    _lazy_fetch_weights from FarabiRobustAggregator, does NOT reuse
    _filter_weight_space_anomalies or Virtue-weighting (FARABI's own
    mechanisms, not part of RFA). Weighting uses each payload's reported
    `samples` count if present (matching Pillutla et al.'s data-size
    weighting), falling back to uniform weights otherwise -- no Byzantine-count
    assumption needed, unlike Multi-Krum/Bulyan.

    See thesis-hub/docs/decisions.md (2026-09-20) for why this baseline was
    added (cross-confirmed in two independent Q1 surveys via the student's
    Zotero library, IEEE Open Journal of the Computer Society and Future
    Generation Computer Systems).
"""

import logging
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class RFAAggregator(FarabiRobustAggregator):
    agg_type = "rfa"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.rfa_iterations = agg_cfg.get("rfa_iterations", 8)
        self.rfa_eps = agg_cfg.get("rfa_eps", 1e-6)
        logger.info(f"⚖️ [RFA] Initialized (Weiszfeld iterations={self.rfa_iterations}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [RFA] Payloads list is empty. Aborting.")
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
            logger.error("❌ [RFA] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights + sample counts via the shared lazy-IPFS-fetch helper ---
        models: List[Dict[str, torch.Tensor]] = []
        sample_weights: List[float] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if not weights:
                continue
            inner = payload.get("payload", payload)
            samples = payload.get("samples") or inner.get("samples") or 1
            models.append(weights)
            sample_weights.append(max(float(samples), 1.0))

        if not models:
            logger.error("❌ [RFA] No valid models fetched. Aborting.")
            return {}

        # Off by default (literature-faithful); on only for the pre-filter
        # ablation. See FarabiRobustAggregator._prefilter_enabled.
        if self._prefilter_enabled(arm_default=False):
            models, sample_weights = self._prefilter_models(models, node, sample_weights)
            if not models:
                logger.error("❌ [RFA] All models dropped by pre-filter. Aborting.")
                return {}

        # Opt-in only (protocols.farabi.aggregation.include_own_state_anchor);
        # default behavior/results unchanged. See _maybe_append_own_state.
        models = self._maybe_append_own_state(models, node, extra_weight_list=sample_weights)

        n = len(models)
        if n == 1:
            return models[0]

        layer_names = [name for name in models[0].keys() if name in models[0]]
        shapes = {name: models[0][name].shape for name in layer_names}
        numels = {name: models[0][name].numel() for name in layer_names}
        dtypes = {name: models[0][name].dtype for name in layer_names}
        is_float = {name: models[0][name].is_floating_point() for name in layer_names}

        flattened = [
            torch.cat([m_[name].flatten().float() for name in layer_names if name in m_])
            for m_ in models
        ]
        weights_t = torch.tensor(sample_weights, dtype=torch.float32)
        weights_t = weights_t / weights_t.sum()

        # Weiszfeld's algorithm: initialize at the weighted mean, then
        # iteratively re-weight each point by the inverse of its current
        # distance to the running estimate.
        v = sum(w * x for w, x in zip(weights_t, flattened))
        for _ in range(self.rfa_iterations):
            dists = torch.tensor([torch.norm(x - v).item() for x in flattened])
            inv_d = weights_t / torch.clamp(dists, min=self.rfa_eps)
            v = sum(iw * x for iw, x in zip(inv_d, flattened)) / inv_d.sum()

        logger.info(f"✅ [RFA] n={n} | geometric median computed over {self.rfa_iterations} Weiszfeld iterations.")

        aggregated: Dict[str, torch.Tensor] = {}
        offset = 0
        for name in layer_names:
            numel = numels[name]
            chunk = v[offset: offset + numel].view(shapes[name])
            offset += numel
            aggregated[name] = chunk if is_float[name] else chunk.round().to(dtypes[name])
        return aggregated
