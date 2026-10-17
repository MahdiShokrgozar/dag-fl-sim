"""
Module: bulyan_aggregator.py
Description:
    Bulyan (El Mhamdi, Guerraoui & Rouault, ICML 2018 -- "The Hidden
    Vulnerability of Distributed Learning in Byzantine Settings") as a
    literature-faithful external baseline for the Farabi-Prox Q1 benchmark.

    Algorithm: recursively run single-Krum selection (each round picks the
    one model with the lowest Krum score among the remaining pool and moves
    it into the selected set) until theta = n - 2f models are selected, then
    apply coordinate-wise trimmed mean over that selected set, trimming the
    f largest and f smallest values per parameter coordinate and averaging
    what remains. This composes two defenses (Krum's distance-based
    selection, then coordinate-wise trimming) specifically so that neither
    alone has to carry the whole robustness burden -- the paper's own
    motivation for why plain Krum or plain trimmed-mean each have exploitable
    blind spots the other does not share.

    Precondition: n >= 4f + 3. At this benchmark's scale (N=10 nodes,
    poisoning fraction 0.2 -> f=2 -> 4f+3=11 > 10), the precondition is NOT
    met -- this is a real, reportable finding about Bulyan's applicability at
    small N, not an implementation bug. When violated, this falls back to
    plain coordinate-wise trimmed-mean (trim=f) with a loud warning log
    instead of silently returning nonsense or crashing; report the fallback
    rate in the paper rather than hiding it.

    Same fetch/filter pipeline reuse rationale as multi_krum_aggregator.py:
    reuses _resolve_aggregation_inputs and _lazy_fetch_weights from
    FarabiRobustAggregator, does NOT reuse _filter_weight_space_anomalies or
    Virtue-weighting (those are FARABI's own mechanisms, not part of Bulyan).
    f is derived the same way as MultiKrumAggregator (see that module's
    docstring for the "assumes an oracle knows the attack rate" caveat).

    See thesis-hub/docs/decisions.md (2026-09-20) for why this baseline was
    added (cross-confirmed in two independent Q1 surveys via the student's
    Zotero library, IEEE Open Journal of the Computer Society and Future
    Generation Computer Systems).
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class BulyanAggregator(FarabiRobustAggregator):
    agg_type = "bulyan"

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

        logger.info(f"⚖️ [BULYAN] Initialized (assumed f={self.assumed_f}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [BULYAN] Payloads list is empty. Aborting.")
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
            logger.error("❌ [BULYAN] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights via the shared lazy-IPFS-fetch helper ---
        models: List[Dict[str, torch.Tensor]] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                models.append(weights)

        if not models:
            logger.error("❌ [BULYAN] No valid models fetched. Aborting.")
            return {}

        # Opt-in only (protocols.farabi.aggregation.include_own_state_anchor);
        # default behavior/results unchanged. See _maybe_append_own_state.
        models = self._maybe_append_own_state(models, node)

        n = len(models)
        f = self.assumed_f
        layer_names = list(models[0].keys())

        if n < 4 * f + 3:
            fallback_trim = min(f, (n - 1) // 2)
            logger.warning(
                f"⚠️ [BULYAN] Precondition n >= 4f+3 violated (n={n}, f={f}, need >= {4*f+3}). "
                f"Bulyan is not well-defined at this scale -- falling back to plain "
                f"coordinate-wise trimmed-mean (trim={fallback_trim}) over all {n} models."
            )
            return self._trimmed_mean(models, list(range(n)), layer_names, trim=fallback_trim)

        theta = n - 2 * f
        selected = self._recursive_krum_select(models, layer_names, theta)
        logger.info(f"✅ [BULYAN] n={n} f={f} theta={theta} | selected indices={selected}")
        return self._trimmed_mean(models, selected, layer_names, trim=f)

    def _recursive_krum_select(
        self, models: List[Dict[str, torch.Tensor]], layer_names: List[str], theta: int
    ) -> List[int]:
        """Repeatedly single-Krum-select the best remaining model until theta are chosen."""
        remaining = list(range(len(models)))
        selected: List[int] = []

        flattened = {
            i: torch.cat([models[i][name].flatten().float() for name in layer_names if name in models[i]])
            for i in remaining
        }

        while len(selected) < theta and len(remaining) > 2:
            k = max(1, len(remaining) - self.assumed_f - 2)
            best_idx, best_score = None, None
            for i in remaining:
                dists = sorted(
                    torch.norm(flattened[i] - flattened[j]).item() ** 2
                    for j in remaining
                    if j != i
                )
                score = sum(dists[:k])
                if best_score is None or score < best_score:
                    best_idx, best_score = i, score
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def _trimmed_mean(
        self,
        models: List[Dict[str, torch.Tensor]],
        selected: List[int],
        layer_names: List[str],
        trim: int,
    ) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        n_sel = len(selected)
        for name in layer_names:
            if name not in models[selected[0]]:
                continue
            stack = torch.stack([models[i][name] for i in selected]).float()
            original_dtype = models[selected[0]][name].dtype
            is_float = models[selected[0]][name].is_floating_point()

            sorted_stack, _ = torch.sort(stack, dim=0)
            if 2 * trim < n_sel:
                trimmed = sorted_stack[trim: n_sel - trim] if trim > 0 else sorted_stack
            else:
                mid = n_sel // 2
                trimmed = sorted_stack[mid: mid + 1]
            result = trimmed.mean(dim=0)
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated
