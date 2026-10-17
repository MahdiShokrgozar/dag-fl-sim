"""
Module: multi_krum_aggregator.py
Description:
    Multi-Krum (Blanchard et al., NeurIPS 2017 -- "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent") as a literature-faithful
    external baseline for the Farabi-Prox Q1 benchmark, wired into this
    protocol's aggregate()-time pipeline.

    Score each model by the sum of squared L2 distances to its k = n-f-2
    nearest neighbors in flattened parameter space (f = assumed max number
    of Byzantine participants), then coordinate-wise average the m = n-f
    models with the lowest (most "central") scores. f=0 -> k=n-2, m=n
    degenerates to plain FedAvg, which is expected and correct.

    Deliberately does NOT reuse FarabiRobustAggregator._filter_weight_space_anomalies
    or its Virtue-weighted classifier-layer averaging: those are FARABI's own
    defenses, and running them ahead of Multi-Krum's own selection would make
    this stop being the literature method it is meant to represent. It DOES
    reuse _resolve_aggregation_inputs (protocol-level DAG anchor resolution,
    not a defense) and _lazy_fetch_weights (IPFS fetch mechanics) plus the
    Genesis/metadata-block filter, exactly like FarabiProxAggregator, to keep
    the weight-fetch/payload-parsing pipeline identical across all baselines
    ("apples-to-apples", see farabi_prox_aggregator.py's module docstring for
    the same argument applied to that aggregator).

    Multi-Krum's own well-known limitation (also true here): it requires
    prior knowledge of f, the max number of Byzantine participants. This
    implementation reads it from `participants.distribution.poisoning *
    participants.total` (ceiling) unless `aggregation.assumed_byzantine_f`
    overrides it -- i.e. it assumes an oracle knows the attack rate, which is
    the same assumption the FedGreed paper (thesis-hub/papers/pdfs/fedGreed.pdf)
    makes explicit for its own Krum/Multi-Krum baselines. Document this
    assumption in the paper; do not present it as free.

    See thesis-hub/docs/decisions.md (2026-09-20) for why this baseline was
    added: two independent Q1-tier surveys (IEEE Open Journal of the Computer
    Society; Future Generation Computer Systems) both cite Multi-Krum as a
    standard robust-aggregation baseline, cross-checked against the student's
    own Zotero library.
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class MultiKrumAggregator(FarabiRobustAggregator):
    agg_type = "multi_krum"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        participants_cfg = self.cfg.get("participants", {})

        explicit_f = agg_cfg.get("assumed_byzantine_f")
        if explicit_f is not None:
            self.assumed_f = int(explicit_f)
        else:
            total = participants_cfg.get("total", 10)
            poisoning_frac = participants_cfg.get("distribution", {}).get("poisoning", 0.0)
            self.assumed_f = math.ceil(total * poisoning_frac)

        logger.info(f"⚖️ [MULTI-KRUM] Initialized (assumed f={self.assumed_f}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [MULTI-KRUM] Payloads list is empty. Aborting.")
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
            logger.error("❌ [MULTI-KRUM] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights via the shared lazy-IPFS-fetch helper ---
        # `authors` is diagnostic-only (added 2026-09-23 to trace WHY Multi-Krum
        # stays collapsed even when its real Krum selection runs with a
        # sufficient population, per thesis-hub/docs/decisions.md 2026-09-23 --
        # plan 32's open question). It does not affect the algorithm's math.
        models: List[Dict[str, torch.Tensor]] = []
        authors: List[str] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                models.append(weights)
                inner = payload.get("payload", payload)
                author = (
                    payload.get("author")
                    or payload.get("proposer")
                    or inner.get("author")
                    or inner.get("proposer")
                    or "unknown"
                )
                authors.append(author)

        if not models:
            logger.error("❌ [MULTI-KRUM] No valid models fetched. Aborting.")
            return {}

        # Opt-in only (protocols.farabi.aggregation.include_own_state_anchor);
        # default behavior/results unchanged. See _maybe_append_own_state.
        models = self._maybe_append_own_state(models, node, extra_weight_list=authors, extra_weight_value="self")

        n = len(models)
        f = min(self.assumed_f, max(0, (n - 3) // 2))  # keep k, m >= 1
        k = max(1, n - f - 2)
        m = max(1, n - f)

        if n < 4:
            logger.warning(
                f"⚠️ [MULTI-KRUM] n={n} too small for meaningful Krum scoring (need >= 4). "
                f"Falling back to plain coordinate-wise mean."
            )
            return self._coordinate_mean(models, list(range(n)))

        layer_names = list(models[0].keys())
        flattened = [
            torch.cat([m_[name].flatten().float() for name in layer_names if name in m_])
            for m_ in models
        ]

        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(flattened[i] - flattened[j]).item() ** 2
                for j in range(n)
                if j != i
            )
            scores.append(sum(dists[:k]))

        ranked = sorted(range(n), key=lambda i: scores[i])
        selected = ranked[:m]
        excluded = ranked[m:]
        # Diagnostic-only (see `authors` comment above): full author-ranked
        # trace, not just the selected subset, so a malicious author's rank
        # and score are visible even when it IS excluded.
        logger.info(
            f"✅ [MULTI-KRUM] n={n} f={f} k={k} m={m} | selected indices={selected} "
            f"scores={[round(scores[i], 2) for i in selected]}"
        )
        logger.info(
            f"🔎 [MULTI-KRUM-TRACE] ranked (author, score): "
            f"{[(authors[i], round(scores[i], 2)) for i in ranked]} | "
            f"excluded_authors={[authors[i] for i in excluded]}"
        )
        return self._coordinate_mean(models, selected)

    def _coordinate_mean(self, models: List[Dict[str, torch.Tensor]], selected: List[int]) -> Dict[str, torch.Tensor]:
        layer_names = list(models[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            stack = torch.stack([models[i][name] for i in selected if name in models[i]])
            original_dtype = models[selected[0]][name].dtype
            is_float = models[selected[0]][name].is_floating_point()
            result = stack.float().mean(dim=0)
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated
