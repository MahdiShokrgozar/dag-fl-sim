"""
Module: balance_aggregator.py
Description:
    BALANCE (Fang, Zhang, Hairi, Khanduri, Liu, Lu, Liu, Gong -- "Byzantine-
    Robust Decentralized Federated Learning", ACM CCS 2024) as a
    literature-faithful external baseline for the Farabi-Prox Q1 benchmark.

    Algorithm (paper's Algorithm 3, Eq. 3-4): each client i uses its OWN
    current intermediate model w_i as a similarity reference. A neighbor's
    model w_j is accepted only if

        ||w_i - w_j|| <= gamma * exp(-kappa * lambda(t)) * ||w_i||          (Eq. 3)

    where lambda(t) in [0, 1] is a monotonically increasing function of
    training progress (the paper's own simplest choice: lambda(t) = t / T,
    reproduced here as elapsed_time / max_time since this simulator is
    time-based/asynchronous rather than round-based -- t/T and
    elapsed/max_time serve the same "how far through training are we"
    role). The accepted set S_i is then blended with the client's own model:

        w_i^{t+1} = alpha * w_i + (1 - alpha) * mean(S_i)                   (Eq. 4)

    Default hyperparameters (alpha=0.5, gamma=0.3, kappa=1.0) are the
    paper's own reported values (Section 6.1.5, "For all datasets, we set
    alpha=0.5, gamma=0.3, kappa=1, lambda(t)=t/T") -- not re-tuned for this
    codebase, to keep the baseline literature-faithful. If S_i is empty
    (every neighbor rejected), the paper's Eq. 4 is undefined (division by
    |S_i|=0); this implementation falls back to pure self-trust
    (w_i^{t+1} = w_i) in that case, the natural reading of "no neighbor was
    close enough to be trusted."

    Deliberately does NOT reuse FarabiRobustAggregator._filter_weight_space_anomalies
    or its Virtue-weighted classifier-layer averaging: those are FARABI's own
    defenses. Reuses _resolve_aggregation_inputs, _lazy_fetch_weights, and the
    Genesis/metadata-block filter, exactly like the other literature
    baselines in this package, to keep the weight-fetch/payload-parsing
    pipeline identical across all baselines ("apples-to-apples").

    Unlike Multi-Krum/Bulyan, BALANCE needs no assumed-Byzantine-count f and
    has no n-vs-f combinatorial precondition -- it is a per-neighbor
    pairwise accept/reject test that works even at n=1. Added per
    thesis-hub/docs/decisions.md (2026-09-23): plan 30's 2x2 control found
    Multi-Krum/Bulyan/RFA collapse only when poisoning is present (not
    non-IID), and plan 31 traced this to Multi-Krum's real precondition
    (n >= 2f+3) and Bulyan's (n >= 4f+3) almost never being met at the small
    per-call populations (usually n<=3-4) this protocol's gossip/DAG tip
    selection produces -- BALANCE's pairwise design sidesteps that
    precondition entirely, which is why it was chosen over further tuning
    of the Krum-family baselines.
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class BalanceAggregator(FarabiRobustAggregator):
    agg_type = "balance"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.alpha = agg_cfg.get("balance_alpha", 0.5)
        self.gamma = agg_cfg.get("balance_gamma", 0.3)
        self.kappa = agg_cfg.get("balance_kappa", 1.0)
        self.max_time = self.cfg.get("simulation", {}).get("max_time", 1.0) or 1.0
        logger.info(
            f"⚖️ [BALANCE] Initialized (alpha={self.alpha}, gamma={self.gamma}, kappa={self.kappa})."
        )

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [BALANCE] Payloads list is empty. Aborting.")
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
            logger.error("❌ [BALANCE] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights via the shared lazy-IPFS-fetch helper ---
        neighbor_models: List[Dict[str, torch.Tensor]] = []
        authors: List[str] = []  # diagnostic only, see the per-merge log line below
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                neighbor_models.append(weights)
                inner = payload.get("payload", payload)
                authors.append(str(payload.get("author") or payload.get("proposer")
                                   or inner.get("author") or inner.get("proposer") or "unknown"))

        if not neighbor_models:
            logger.error("❌ [BALANCE] No valid neighbor models fetched. Aborting.")
            return {}

        # Off by default (literature-faithful); on only for the pre-filter
        # ablation. See FarabiRobustAggregator._prefilter_enabled.
        if self._prefilter_enabled(arm_default=False):
            (neighbor_models,) = self._prefilter_models(neighbor_models, node)
            authors = ["?"] * len(neighbor_models)  # the filter does not report which it dropped
            if not neighbor_models:
                logger.error("❌ [BALANCE] All models dropped by pre-filter. Aborting.")
                return {}

        # --- Stage 2: own model as similarity reference (the whole point of BALANCE) ---
        own_weights = None
        if node is not None and hasattr(node, "weights") and node.weights:
            own_weights = node.weights

        if own_weights is None:
            # No local reference yet (e.g. very first round) -- cannot run
            # Eq. 3 at all. Fall back to plain coordinate-wise mean over the
            # neighbor set, same degenerate-case spirit as the other
            # baselines' n<4/precondition fallbacks.
            logger.warning(
                "⚠️ [BALANCE] node.weights unavailable -- no similarity reference. "
                "Falling back to plain coordinate-wise mean over neighbors."
            )
            return self._coordinate_mean(neighbor_models)

        layer_names = [name for name in own_weights.keys()]
        flat_own = torch.cat([own_weights[name].flatten().float() for name in layer_names if name in own_weights])
        own_norm = torch.norm(flat_own).item()

        current_time = 0.0
        if node is not None:
            current_time = getattr(getattr(node, "simulator", None), "current_time", 0.0) or 0.0
        lam = min(max(current_time / self.max_time, 0.0), 1.0)
        threshold = self.gamma * math.exp(-self.kappa * lam) * own_norm

        accepted: List[Dict[str, torch.Tensor]] = []
        dists: List[float] = []
        for model in neighbor_models:
            flat_j = torch.cat([model[name].flatten().float() for name in layer_names if name in model])
            dist = torch.norm(flat_own - flat_j).item()
            dists.append(dist)
            if dist <= threshold:
                accepted.append(model)
        # Diagnostic only (thesis-hub/docs/decisions.md, 2026-09-30, second Q1
        # review, item N1): which parents the threshold rejected, so that the
        # rejection of honest parents can be counted from the log.
        logger.info(
            f"🧮 [BALANCE] merge authors={','.join(authors)} "
            f"d={','.join(f'{d:.2f}' for d in dists)} "
            f"acc={','.join('1' if d <= threshold else '0' for d in dists)} threshold={threshold:.4f}"
        )

        logger.info(
            f"✅ [BALANCE] n_neighbors={len(neighbor_models)} accepted={len(accepted)} "
            f"lambda(t)={lam:.3f} threshold={threshold:.4f} own_norm={own_norm:.4f}"
        )

        # --- Stage 3: Eq. 4 blend (own model dominates when S_i is empty) ---
        aggregated: Dict[str, torch.Tensor] = {}
        if not accepted:
            logger.warning("⚠️ [BALANCE] All neighbors rejected -- pure self-trust this round.")
            for name in layer_names:
                aggregated[name] = own_weights[name]
            return aggregated

        neighbor_mean = self._coordinate_mean(accepted)
        for name in layer_names:
            if name not in own_weights or name not in neighbor_mean:
                continue
            own_val = own_weights[name].float()
            neigh_val = neighbor_mean[name].float()
            original_dtype = own_weights[name].dtype
            is_float = own_weights[name].is_floating_point()
            result = self.alpha * own_val + (1 - self.alpha) * neigh_val
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated

    def _coordinate_mean(self, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        layer_names = list(models[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            stack = torch.stack([m[name] for m in models if name in m]).float()
            original_dtype = models[0][name].dtype
            is_float = models[0][name].is_floating_point()
            result = stack.mean(dim=0)
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated
