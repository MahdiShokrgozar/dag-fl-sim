"""
Module: farabi_global_prox_aggregator.py
Description:
    Global-anchored proximal aggregator for the FARABI protocol. A one-variable
    ablation of FarabiProxAggregator: identical aggregate()-time exponential
    proximal-dampening + Virtue-weighting math,

        Weight_i = Virtue_i * exp(-mu * ||W_i - W_anchor||_2) * exp(-lambda * age_i) / sum(...)

    but the anchor is switched from "highest-Virtue DAG parent in this
    round's payload set" (FarabiProxAggregator) to "this node's own
    pre-round model state" (`node.weights`).

    WHY THIS EXISTS — NOT A FedProx REIMPLEMENTATION, DO NOT CALL IT THAT.
    We wanted a literature FedProx baseline for the Farabi-Prox Q1 paper.
    The real FedProx class (src/strategies/aggregation/fedprox.py) applies
    its proximal penalty inside client_update() during local SGD, anchored
    to the previous synchronized global model. But client_update() on
    aggregator objects is only ever invoked by the centralized engine
    (src/simulators/engines/centralized.py) -- the l2_scaling/DAG backend
    that every FARABI experiment uses calls only aggregate(), never
    client_update(). Wiring the real FedProx class in under that backend
    would silently run plain averaging (its aggregate() has no proximal
    term at all) while the results table said "FedProx" -- a correctness
    bug, not a baseline. See thesis-hub/docs/decisions.md, 2026-09-19.

    This class is the deliberate alternative: it moves the proximal penalty
    to aggregate()-time (where this DAG architecture actually has a hook
    for it, same as FarabiProxAggregator already does) instead of trying to
    force a training-time mechanism into a backend that doesn't call it.
    There is also no single synchronized "global model" in this async-DAG
    setting the way round-based FL has one, so "the previous global
    checkpoint" is approximated by `node.weights` -- each node's own
    most-recently-accepted local model state, i.e. what it held immediately
    before this aggregation round. This is not a new concept introduced
    here: `node.weights` is the same "local trust anchor" already relied on
    by FarabiRobustAggregator._filter_weight_space_anomalies (see that
    method's Case 1/2/3 in farabi_aggregator.py) for exactly the same
    "what did this node already trust" role.

    Report both facts plainly in the paper's Methods section: (1) the
    anchor-choice ablation this ARM vs. farabi_prox isolates (local DAG
    parent vs. own prior state), and (2) that this is a DAG-consistent
    proximal analog, not a reproduction of textbook FedProx's training-time
    mechanism -- the two are not claimed to be equivalent.

    Subclasses FarabiRobustAggregator and reuses its Genesis/metadata-block
    filtering, `_lazy_fetch_weights`, and `_filter_weight_space_anomalies`
    exactly as FarabiProxAggregator does, for the same reason: keep the
    three-way benchmark (robust / prox / global_prox) apples-to-apples on
    everything except the one thing each is meant to isolate.
"""

import logging
import math
from typing import Any, Dict, List

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class FarabiGlobalProxAggregator(FarabiRobustAggregator):
    agg_type = "farabi_global_prox"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        # Same config key as FarabiProxAggregator's proximal_mu (configs/
        # defaults.yaml: protocols.farabi.aggregation.proximal_mu: 0.5).
        # Sharing the key is safe: a run only ever selects one aggregation
        # strategy at a time, never both in the same variation, and using
        # the same key means both arms are compared at the project's one
        # already-calibrated mu rather than two independently-guessed ones.
        self.proximal_mu = (
            self.cfg.get("protocols", {})
            .get("farabi", {})
            .get("aggregation", {})
            .get("proximal_mu", 2.0)
        )
        # Staleness weight, same config block, age unit and formula as
        # FarabiProxAggregator (thesis-hub/docs/decisions.md, 2026-09-28): the
        # two arms are one mechanism with two anchor choices, so they must
        # differ only in the anchor. Plans 50/53 ran this arm without it;
        # lambda_aggregate: 0 reproduces that behaviour.
        st_cfg = (
            self.cfg.get("protocols", {}).get("farabi", {})
            .get("selection", {}).get("staleness", {}) or {}
        )
        self.staleness_enabled = bool(st_cfg.get("enabled", False))
        self.staleness_lambda_aggregate = float(st_cfg.get("lambda_aggregate", 0.5))
        # Reputation factor on/off (thesis-hub/docs/decisions.md, 2026-09-29,
        # reviewer item M5): false replaces every proposer's Virtue by the same
        # constant, isolating what reputation contributes to the merge. Default
        # true = unchanged behaviour.
        self.reputation_weight = bool(
            self.cfg.get("protocols", {}).get("farabi", {})
            .get("aggregation", {}).get("reputation_weight", True)
        )
        # What to do when every parent's weight underflows to exactly 0.0
        # (thesis-hub/docs/decisions.md, 2026-09-29, reviewer item M3):
        #   "uniform" (default, every reported result) - average the parents
        #       uniformly; if they are all poisoned, the trainer's divergence
        #       guard rejects that average and the node trains from its own model;
        #   "own" - keep the node's own model. NOT equivalent: poisoners run this
        #       aggregator too, and under "own" they keep adding noise to their
        #       own previous noise, which weakens the attack (25-node check).
        self.zero_weight_fallback = str(
            self.cfg.get("protocols", {}).get("farabi", {})
            .get("aggregation", {}).get("zero_weight_fallback", "uniform")
        )
        # Shape of the distance factor (thesis-hub/docs/decisions.md, 2026-09-30,
        # second Q1 review, item N1): "exp" (default, every reported result) is
        # exp(-mu * d); "hard" replaces it by 1 if d <= hard_radius else 0, with
        # reputation, age and the zero-weight fallback unchanged. The "hard" arm
        # is an ablation that asks whether the continuous kernel matters or only
        # the self anchor does; it is not part of the method.
        agg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.distance_kernel = str(agg.get("distance_kernel", "exp"))
        if self.distance_kernel not in ("exp", "hard"):
            raise ValueError(f"distance_kernel must be 'exp' or 'hard', got {self.distance_kernel!r}")
        self.hard_radius = float(agg.get("hard_radius", 4.0))
        logger.info(
            f"⚖️ [FARABI-GLOBAL-PROX] Initialized (mu={self.proximal_mu}, kernel={self.distance_kernel}"
            + (f", radius={self.hard_radius}" if self.distance_kernel == "hard" else "") + ")."
        )

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [FARABI-GLOBAL-PROX] Payloads list is empty. Aborting.")
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
            logger.error("❌ [FARABI-GLOBAL-PROX] All payloads were empty/Genesis. Aborting.")
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
            fetched.append({"node_id": author, "weights": weights, "timestamp": payload.get("timestamp")})

        if not fetched:
            logger.error("❌ [FARABI-GLOBAL-PROX] No valid models fetched. Aborting.")
            return {}

        # --- Stage 1b: weight-space anomaly filter (inherited from FarabiRobustAggregator; see
        # FarabiProxAggregator's identical stage for why this must not be skipped) ---
        # Off by default since 2026-09-28 (thesis-hub/docs/decisions.md, plan 56):
        # for this arm the filter never changed a result - the distance damping
        # already gives every parent it would drop a weight of exactly 0.0 - so
        # it is not part of the method. prefilter: true restores it.
        if self._prefilter_enabled(arm_default=False):
            fetched = self._filter_weight_space_anomalies(fetched, node)
        if not fetched:
            logger.error("❌ [FARABI-GLOBAL-PROX] All models dropped by anomaly filter. Aborting.")
            return {}

        models_to_aggregate: List[Dict[str, torch.Tensor]] = []
        reputation_scores: List[float] = []
        authors: List[str] = []
        staleness: List[float] = []
        ages: List[float] = []
        period = getattr(node, "staleness_round_period", None) if node else None
        now = float(getattr(getattr(node, "simulator", None), "current_time", 0.0) or 0.0)
        for item in fetched:
            author = item["node_id"]
            virtue = 50.0
            if self.reputation_weight and node and hasattr(node, "reputation"):
                virtue = node.reputation.get_virtue_score(author)
            virtue = max(virtue, 0.1)  # prevent zero division

            models_to_aggregate.append(item["weights"])
            reputation_scores.append(virtue)
            authors.append(author)
            factor = 1.0
            ts = item.get("timestamp")
            if self.staleness_enabled and period and ts is not None:
                age_rounds = max(0.0, now - float(ts)) / period
                factor = math.exp(-self.staleness_lambda_aggregate * age_rounds)
                ages.append(age_rounds)
            staleness.append(factor)
        if ages:
            # Diagnostic only (plan 57): parent-age spread per merge, the quantity
            # the aggregation-side factor depends on - with near-equal ages the
            # factor cancels in the normalization below.
            logger.info(f"⏳ [FARABI-GLOBAL-PROX] parent ages (rounds): {' '.join(f'{a:.2f}' for a in ages)}")

        # --- Stage 2: anchor selection -- this node's own pre-round state, not the
        # highest-virtue fetched parent (that substitution is the whole point of this
        # class; see module docstring). Cold-start fallback: if this node has no prior
        # weights yet (e.g. very first aggregation it ever performs), there is nothing
        # to anchor to -- fall back to the highest-virtue fetched model, i.e. the same
        # anchor FarabiProxAggregator would pick, so this arm degrades to that one
        # exactly once per node's lifetime instead of crashing or dividing by an empty
        # anchor.
        local_weights = getattr(node, "weights", None) if node else None
        if local_weights:
            anchor_model = local_weights
            anchor_label = "own pre-round state"
        else:
            anchor_idx = reputation_scores.index(max(reputation_scores))
            anchor_model = models_to_aggregate[anchor_idx]
            anchor_label = f"cold-start fallback: {authors[anchor_idx]} (highest virtue)"
        logger.info(f"⚓ [FARABI-GLOBAL-PROX] Anchor: {anchor_label}")

        # --- Stage 3: L2 distance to anchor + proximal-dampened effective weight ---
        effective_weights = []
        distances = []
        for i, model in enumerate(models_to_aggregate):
            dist_sq = sum(
                (model[k] - anchor_model[k]).float().norm().item() ** 2
                for k in model
                if k in anchor_model
            )
            dist = math.sqrt(dist_sq)
            distances.append(dist)

            if self.distance_kernel == "hard":
                prox_factor = 1.0 if dist <= self.hard_radius else 0.0
            else:
                prox_factor = math.exp(-self.proximal_mu * dist)
            effective_weights.append(reputation_scores[i] * prox_factor * staleness[i])

        total_eff_weight = sum(effective_weights)
        if total_eff_weight <= 0:
            # Diagnostic (reviewer item M3): how often every parent is rejected.
            logger.info(
                f"🛑 [FARABI-GLOBAL-PROX] all {len(effective_weights)} parents at zero "
                f"weight (d={','.join(f'{d:.1f}' for d in distances)}); "
                f"fallback={self.zero_weight_fallback}."
            )
        if total_eff_weight <= 0 and local_weights and self.zero_weight_fallback == "own":
            return {k: v.clone() if torch.is_tensor(v) else v for k, v in local_weights.items()}
        if total_eff_weight <= 0:
            normalized_weights = [1.0 / len(effective_weights)] * len(effective_weights)
        else:
            normalized_weights = [w / total_eff_weight for w in effective_weights]
        # Diagnostic only (reviewer item M4): per-merge distances to the anchor,
        # normalized weights and effective number of parents 1 / sum(w^2).
        n_eff = 1.0 / sum(w * w for w in normalized_weights)
        logger.info(
            f"🧮 [FARABI-GLOBAL-PROX] merge authors={','.join(authors)} "
            f"d={','.join(f'{d:.2f}' for d in distances)} "
            f"w={','.join(f'{w:.4f}' for w in normalized_weights)} n_eff={n_eff:.3f}"
        )

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

        logger.info(f"✅ [FARABI-GLOBAL-PROX] Aggregated {len(models_to_aggregate)} models.")
        return aggregated_weights
