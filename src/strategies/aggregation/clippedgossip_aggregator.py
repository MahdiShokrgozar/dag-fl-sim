"""
Module: clippedgossip_aggregator.py
Description:
    ClippedGossip (He, Karimireddy & Jaggi, "Byzantine-Robust Decentralized
    Learning via ClippedGossip", arXiv:2202.01545) as a literature baseline
    for the Farabi-Prox Q1 benchmark (reviewer item M1; thesis-hub
    docs/decisions.md 2026-09-30).

    Update rule (paper, Sec. 4.1), with the node's own model x_i as center:
        x_i <- sum_j W_ij (x_i + CLIP(x_j - x_i, tau_i)),
        CLIP(z, tau) = min(1, tau / ||z||) * z
    Adaptive clipping radius (paper, App. D.5, Eq. 10): sort the neighbours by
    ||x_i - x_j||^2, take the closest set S whose mixing weights sum to at most
    1 - delta_max, and set
        tau_i = sqrt( sum_{j in S} W_ij ||x_i - x_j||^2 ).

    Mapping onto a DAG merge: the neighbours are the parents chosen by tip
    selection, and the node itself is a member of the mixing set (W_ii > 0, as
    in the paper's gossip matrices). With no other information the mixing
    weights are uniform over self + parents, W = 1/(n+1). delta_max is the
    assumed Byzantine weight in a neighbourhood (config
    `clippedgossip_delta`, default 0.2 = the attacker share of our scenarios):
    like f for Krum/Bulyan, this is knowledge the baseline is given, not
    something Farabi-Prox needs.

    Cold start (the node has no model yet, first merge of its lifetime): the
    center is the parent with the smallest sum of distances to the others (the
    medoid), so the rule still clips around a parent chosen without reputation.

    Same fetch pipeline reuse as rfa_aggregator.py: _resolve_aggregation_inputs
    and _lazy_fetch_weights from FarabiRobustAggregator; no FARABI pre-filter
    and no reputation weighting (not part of ClippedGossip).
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


def adaptive_tau(sq_dists: List[float], weights: List[float], delta: float) -> float:
    """Eq. 10 of the paper: closest set with total weight <= 1 - delta."""
    order = sorted(range(len(sq_dists)), key=lambda j: sq_dists[j])
    budget, acc, s = 1.0 - delta, 0.0, 0.0
    for j in order:
        if acc + weights[j] > budget + 1e-12:
            break
        acc += weights[j]
        s += weights[j] * sq_dists[j]
    return math.sqrt(s)


def clipped_gossip(center: torch.Tensor, neighbours: List[torch.Tensor], delta: float):
    """One ClippedGossip step for flattened models. Returns (new_center, tau, n_clipped).

    The mixing set is self + neighbours with uniform weights 1/(n+1); the self
    term contributes x_i + CLIP(0) = x_i.
    """
    n = len(neighbours)
    w = 1.0 / (n + 1)
    diffs = [x - center for x in neighbours]
    norms = [d.norm().item() for d in diffs]
    tau = adaptive_tau([0.0] + [v * v for v in norms], [w] * (n + 1), delta)
    out = center.clone()
    n_clipped = 0
    for d, nrm in zip(diffs, norms):
        scale = 1.0 if nrm <= tau or nrm == 0.0 else tau / nrm
        n_clipped += scale < 1.0
        out = out + w * scale * d
    return out, tau, n_clipped


class ClippedGossipAggregator(FarabiRobustAggregator):
    agg_type = "clippedgossip"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.delta = float(agg_cfg.get("clippedgossip_delta", 0.2))
        logger.info(f"⚖️ [CLIPPEDGOSSIP] Initialized (adaptive tau, delta_max={self.delta}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)
        if not payloads:
            logger.error("❌ [CLIPPEDGOSSIP] Payloads list is empty. Aborting.")
            return {}

        # --- Stage 0: drop empty/Genesis/metadata-only blocks (same rule as RFA) ---
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
            if not is_genesis:
                valid_blocks.append(p)
        if not valid_blocks:
            logger.error("❌ [CLIPPEDGOSSIP] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch parent weights ---
        models: List[Dict[str, torch.Tensor]] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                models.append(weights)
        if not models:
            logger.error("❌ [CLIPPEDGOSSIP] No valid models fetched. Aborting.")
            return {}

        layer_names = list(models[0].keys())
        shapes = {k: models[0][k].shape for k in layer_names}
        dtypes = {k: models[0][k].dtype for k in layer_names}
        is_float = {k: models[0][k].is_floating_point() for k in layer_names}
        flat = lambda m: torch.cat([m[k].flatten().float() for k in layer_names])  # noqa: E731
        parents = [flat(m) for m in models]

        # --- Stage 2: center = the node's own model (cold start: medoid parent) ---
        own = getattr(node, "weights", None) if node else None
        if own and all(k in own for k in layer_names):
            center, label = flat(own), "own model"
        else:
            sums = [sum((a - b).norm().item() for b in parents) for a in parents]
            i = sums.index(min(sums))
            center, label = parents[i], f"cold-start medoid parent {i}"

        v, tau, n_clipped = clipped_gossip(center, parents, self.delta)
        logger.info(
            f"✅ [CLIPPEDGOSSIP] n={len(parents)} | center={label} | tau={tau:.4f} | clipped={n_clipped}"
        )

        aggregated: Dict[str, torch.Tensor] = {}
        offset = 0
        for k in layer_names:
            numel = models[0][k].numel()
            chunk = v[offset: offset + numel].view(shapes[k])
            offset += numel
            aggregated[k] = chunk if is_float[k] else chunk.round().to(dtypes[k])
        return aggregated
