"""
Module: wfagg_aggregator.py
Description:
    WFAgg (Cajaraville-Aboy, Fernandez-Vilas, Diaz-Redondo, Fernandez-Veiga --
    "Byzantine-Robust Aggregation for Securing Decentralized Federated
    Learning", IEEE Access 2025, arXiv:2409.17754) as a literature-faithful
    external baseline for the Q1 decentralized-aggregation benchmark.

    Three parallel filters, each producing an accept/reject verdict per
    neighbor model, run against the current round's population:

    - WFAgg-D (distance): theta_Med = coordinate-wise median of all K
      received models. d_j = ||theta_j - theta_Med||^2. T1 = the (K-f-1)
      models with smallest d_j.
    - WFAgg-C (cosine): theta_Med as above; tau_Med = median of model norms;
      norm-clip each theta_j toward tau_Med (theta'_j = theta_j *
      min(1, tau_Med/||theta_j||)); alpha_j = 1 - cos(angle(theta'_j,
      theta_Med)). T2 = the (K-f-1) models with smallest alpha_j.
    - WFAgg-T (temporal): inert until this neighbor has a full window W of
      history. Per neighbor j, an EWMA mean/std over its own last W rounds'
      round-to-round deltas (distance and cosine, computed against THAT
      neighbor's own previous model, not theta_Med). j survives only if
      this round's delta falls within [mean-std, mean+std] on BOTH metrics.

    Combine (paper's Algorithm 1): w_j = tau1*[j in T1] + tau2*[j in T2] +
    tau3*[j in T3] (tau1=tau2=0.4, tau3=0.2 -- paper's own values, Sec V-A);
    a model survives only if it is accepted by >= 2 of the 3 filters (i.e.
    its raw filter-membership count >= 2 -- implemented as an explicit count
    rather than thresholding the tau-weighted sum, which is equivalent for
    these tau values and avoids float-threshold edge cases).

    Aggregation (WFAgg-E, self-anchored): theta_bar = (1-alpha)*theta_self +
    alpha * sum(w'_j * theta_j), w' = w normalized to sum to 1 over
    survivors, alpha=0.8 (paper's own value).

    STATE ACROSS ROUNDS: this is the one baseline in this package that needs
    it. protocols.farabi.FarabiProtocol.get_aggregator() constructs exactly
    one aggregator instance per node (see src/nodes/factory.py), and that
    instance lives for the node's whole run -- so a plain dict on
    `self` (keyed by author id, the same identity Multi-Krum's trace log
    uses) persists correctly across this node's successive aggregate()
    calls. No cross-node/cross-run persistence is needed or attempted.

    ADAPTATION TO THIS SIMULATOR (disclose in the paper): the paper is
    round-based ("t > T_th" rounds); this protocol is async/DAG-based with
    no global round counter reaching every aggregator uniformly. WFAgg-T is
    therefore gated PER NEIGHBOR on "this neighbor's own history buffer is
    full (length == W)" rather than on a shared global round index -- the
    same "how far into training are we, locally" substitution BALANCE makes
    for its round-based lambda(t) (see balance_aggregator.py's docstring).

    Deliberately does NOT reuse FarabiRobustAggregator._filter_weight_space_anomalies
    or its Virtue-weighted classifier-layer averaging. Reuses
    _resolve_aggregation_inputs, _lazy_fetch_weights, the Genesis/metadata
    filter, and (optionally) _maybe_append_own_state, exactly like the other
    literature baselines in this package.

    Oracle assumption (same limitation as Multi-Krum, disclose in the
    paper): f (assumed max Byzantine count) is read from
    participants.distribution.poisoning * participants.total (ceiling)
    unless aggregation.assumed_byzantine_f overrides it.

    ⚠️ VALIDATION-DOMAIN CAVEAT (disclose in the paper, do not present WFAgg
    as validated for this setting): the paper's own experiments (Sec V-A)
    are explicitly IID-only -- "non-IID settings introduce additional
    convergence variability unrelated to the focus of this work." Running
    it under this repo's Dirichlet non-IID + poisoning scenarios is new,
    unreported territory for this method, not a reproduction of a validated
    result. See thesis-hub/docs/decisions.md (2026-09-27) for why WFAgg was
    added despite this -- it was one of the three "decentralized/P2P,
    genuinely no trusted server" candidates identified in the 2026-09-23
    literature review that had not yet been implemented (BALANCE, the
    fourth, was implemented and tested in plan 50).
"""

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


class WFAggAggregator(FarabiRobustAggregator):
    agg_type = "wfagg"

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

        self.tau1 = float(agg_cfg.get("wfagg_tau1", 0.4))
        self.tau2 = float(agg_cfg.get("wfagg_tau2", 0.4))
        self.tau3 = float(agg_cfg.get("wfagg_tau3", 0.2))
        self.min_filters_to_survive = int(agg_cfg.get("wfagg_min_filters", 2))
        self.alpha = float(agg_cfg.get("wfagg_alpha", 0.8))
        self.window = int(agg_cfg.get("wfagg_window", 3))  # W = T_th in the paper

        # Per-neighbor persistent state: author_id -> {"last_model": flat tensor,
        # "dist_hist": deque, "cos_hist": deque}. Lives as long as this
        # aggregator instance (i.e. this node) does -- see module docstring.
        self._history: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"⚖️ [WFAGG] Initialized (assumed f={self.assumed_f}, "
            f"tau=({self.tau1},{self.tau2},{self.tau3}), alpha={self.alpha}, W={self.window})."
        )

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [WFAGG] Payloads list is empty. Aborting.")
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
            logger.error("❌ [WFAGG] All payloads were empty/Genesis. Aborting.")
            return {}

        # --- Stage 1: fetch weights + author id ---
        models: List[Dict[str, torch.Tensor]] = []
        authors: List[str] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                inner = payload.get("payload", payload)
                author = (
                    payload.get("author")
                    or payload.get("proposer")
                    or inner.get("author")
                    or inner.get("proposer")
                    or "unknown"
                )
                models.append(weights)
                authors.append(author)

        if not models:
            logger.error("❌ [WFAGG] No valid models fetched. Aborting.")
            return {}

        own_weights = None
        if node is not None and hasattr(node, "weights") and node.weights:
            own_weights = node.weights

        if own_weights is None:
            logger.warning(
                "⚠️ [WFAGG] node.weights unavailable -- WFAgg-E has no self-anchor. "
                "Falling back to plain coordinate-wise mean over neighbors."
            )
            return self._coordinate_mean(models, list(range(len(models))))

        layer_names = list(own_weights.keys())
        flat = [
            torch.cat([m[name].flatten().float() for name in layer_names if name in m])
            for m in models
        ]
        K = len(flat)

        if K < 3:
            logger.warning(
                f"⚠️ [WFAGG] K={K} too small for median-based filters (need >= 3). "
                f"Falling back to WFAgg-E self-anchor over the raw population."
            )
            return self._self_anchor_blend(own_weights, models, list(range(K)), layer_names)

        f = min(self.assumed_f, max(0, (K - 2) // 2))
        keep_n = max(1, K - f - 1)

        # --- WFAgg-D: distance-to-median filter ---
        theta_med = torch.stack(flat).median(dim=0).values
        d = torch.stack([torch.norm(v - theta_med) ** 2 for v in flat])
        t1 = set(torch.argsort(d)[:keep_n].tolist())

        # --- WFAgg-C: norm-clipped cosine-to-median filter ---
        norms = torch.stack([torch.norm(v) for v in flat])
        tau_med = norms.median()
        clipped = [
            v * torch.clamp(tau_med / (torch.norm(v) + 1e-12), max=1.0) for v in flat
        ]
        cos_to_med = torch.stack(
            [torch.nn.functional.cosine_similarity(v.unsqueeze(0), theta_med.unsqueeze(0)).squeeze(0) for v in clipped]
        )
        alpha_dist = 1.0 - cos_to_med
        t2 = set(torch.argsort(alpha_dist)[:keep_n].tolist())

        # --- WFAgg-T: per-neighbor temporal consistency (stateful) ---
        t3 = set()
        for i in range(K):
            author = authors[i]
            hist = self._history.get(author)
            cur_flat = flat[i]
            if hist is not None and hist["dist_hist"].maxlen == self.window:
                s_j = torch.norm(cur_flat - hist["last_model"]).item() ** 2
                beta_j = 1.0 - torch.nn.functional.cosine_similarity(
                    cur_flat.unsqueeze(0), hist["last_model"].unsqueeze(0)
                ).item()
                if len(hist["dist_hist"]) == self.window:
                    mu_d, sd_d = _mean_std(hist["dist_hist"])
                    mu_c, sd_c = _mean_std(hist["cos_hist"])
                    if (mu_d - sd_d) <= s_j <= (mu_d + sd_d) and (mu_c - sd_c) <= beta_j <= (mu_c + sd_c):
                        t3.add(i)
                hist["dist_hist"].append(s_j)
                hist["cos_hist"].append(beta_j)
                hist["last_model"] = cur_flat.clone()
            else:
                self._history[author] = {
                    "last_model": cur_flat.clone(),
                    "dist_hist": deque(maxlen=self.window),
                    "cos_hist": deque(maxlen=self.window),
                }
                # First sighting of this neighbor: no delta to compute yet,
                # WFAgg-T abstains (neither accepts nor rejects) until a
                # history exists -- matches the paper's "inert before T_th".

        # --- Combine: survive only with >= min_filters_to_survive votes ---
        weights_i = []
        survivors = []
        for i in range(K):
            votes = int(i in t1) + int(i in t2) + int(i in t3)
            if votes >= self.min_filters_to_survive:
                w = (
                    (self.tau1 if i in t1 else 0.0)
                    + (self.tau2 if i in t2 else 0.0)
                    + (self.tau3 if i in t3 else 0.0)
                )
                survivors.append(i)
                weights_i.append(w)

        logger.info(
            f"✅ [WFAGG] K={K} f={f} |T1|={len(t1)} |T2|={len(t2)} |T3|={len(t3)} "
            f"survivors={len(survivors)} authors_kept={[authors[i] for i in survivors]}"
        )

        if not survivors:
            logger.warning("⚠️ [WFAGG] No model survived >=2-of-3 filters -- pure self-trust this round.")
            return {name: own_weights[name] for name in layer_names}

        return self._self_anchor_blend(own_weights, models, survivors, layer_names, weights_i)

    def _self_anchor_blend(
        self,
        own_weights: Dict[str, torch.Tensor],
        models: List[Dict[str, torch.Tensor]],
        survivor_indices: List[int],
        layer_names: List[str],
        survivor_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """WFAgg-E: theta_bar = (1-alpha)*theta_self + alpha * weighted_mean(survivors)."""
        if survivor_weights is None:
            survivor_weights = [1.0] * len(survivor_indices)
        total = sum(survivor_weights) or 1.0
        norm_w = [w / total for w in survivor_weights]

        aggregated: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            if name not in own_weights:
                continue
            candidate_layers = [models[i].get(name) for i in survivor_indices if name in models[i]]
            if not candidate_layers:
                aggregated[name] = own_weights[name]
                continue
            usable_w = [
                norm_w[j] for j, i in enumerate(survivor_indices) if name in models[i]
            ]
            usable_w_sum = sum(usable_w) or 1.0
            stack = torch.stack([t.float() for t in candidate_layers])
            w_tensor = torch.tensor(usable_w, dtype=torch.float32).view(-1, *([1] * (stack.dim() - 1)))
            neighbor_mean = (stack * (w_tensor / usable_w_sum)).sum(dim=0)

            own_val = own_weights[name].float()
            original_dtype = own_weights[name].dtype
            is_float = own_weights[name].is_floating_point()
            result = (1.0 - self.alpha) * own_val + self.alpha * neighbor_mean
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated

    def _coordinate_mean(self, models: List[Dict[str, torch.Tensor]], indices: List[int]) -> Dict[str, torch.Tensor]:
        layer_names = list(models[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            stack = torch.stack([models[i][name] for i in indices if name in models[i]]).float()
            original_dtype = models[indices[0]][name].dtype
            is_float = models[indices[0]][name].is_floating_point()
            result = stack.mean(dim=0)
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated


def _mean_std(values: "deque") -> Tuple[float, float]:
    xs = torch.tensor(list(values), dtype=torch.float32)
    return xs.mean().item(), (xs.std(unbiased=False).item() or 1e-6)
