"""
Module: dfl_dual_aggregator.py
Description:
    "DFL-Dual" (Sun, Li, Wang, ... "Byzantine-robust Decentralized Federated
    Learning via Dual-domain Clustering and Trust Bootstrapping", CVPR 2024)
    as a literature-faithful external baseline for the Q1 decentralized-
    aggregation benchmark.

    Algorithm (this file's own reconstruction from the 10-page main text --
    the supplementary material with exact alpha/C1/C2/DLG-iteration values
    was NOT available when this was written; every numeric default below is
    OUR OWN choice, disclosed, not the paper's):

    1. Population P for this round = {own current model} U {fetched
       neighbor models}. Own model is index 0.
    2. Data-domain distance: synthesize a small per-model "dummy dataset"
       for every NEIGHBOR model (own model uses a small real local-data
       sample instead of an inversion -- see the "OWN MODEL" note below).
       Pairwise Wasserstein distance W_kj between two models' dummy sets.
    3. Model-domain distance: pairwise Euclidean distance E_kj between the
       flattened weight vectors.
    4. Dual-domain distance: D_kj = min(W_kj + alpha*E_kj, C1).
    5. Trust score of model k w.r.t. THIS node (i): S_k = cosine similarity
       between k's model and the node's own current model (index 0).
    6. Stage 1 (per neighbor j): 2-median-cluster the REST of the population
       (everyone except j) by their D-to-j distance; keep the higher-
       average-trust-score cluster M*_j; q_j = sum(D over M*_j).
       q_own is computed the same way using "own model" as the reference
       column instead of j.
    7. Divergence rate: r_j = min(q_j / q_own, C2).
    8. Stage 2: 2-median-cluster all neighbors by r_j; the higher-trust
       cluster N*_i survives. Aggregate = plain coordinate-wise mean over
       N*_i (paper's Eq. 3 -- no self-anchor blend at this final step,
       unlike BALANCE/WFAgg/LIGHTYEAR).
    9. NO STATE ACROSS ROUNDS -- everything above is recomputed from
       scratch every aggregate() call, unlike WFAgg.

    OWN MODEL / "MODEL INVERSION" SUBSTITUTION (disclose prominently in the
    paper -- this is the single biggest documented deviation from the
    source method): the paper's own text calls this "privacy-respecting
    model inversion" in the DLG (Deep Leakage from Gradients) family, but
    classic DLG needs an intercepted per-step GRADIENT to invert, and this
    protocol only ever transmits final post-training WEIGHTS, never a
    captured gradient. What is actually implemented here is a class-
    representative synthesis: for a fixed neighbor model theta_j (frozen,
    requires_grad_=False), optimize `dfl_dual_samples_per_class` random
    images per class via gradient DESCENT on the cross-entropy loss toward
    that target class under theta_j (i.e. "what input would theta_j most
    confidently call class c"), for `dfl_dual_inversion_iters` Adam steps.
    This is the closest feasible analogue to the paper's mechanism given
    what this protocol actually exchanges, not a literal reimplementation
    of DLG's gradient-matching objective -- and it is the dominant cost
    driver of this baseline (see the module-level cost-gate note in
    thesis-hub/docs/decisions.md, 2026-09-27). For the node's OWN model
    (index 0) we skip inversion entirely and instead draw a small sample of
    that node's genuine local training data -- the node already has real
    data for itself, so inverting its own model would be needless extra
    cost for no more information.

    WASSERSTEIN APPROXIMATION (disclosed simplification, not the paper's
    exact image-level optimal-transport formulation): W_kj is computed as
    the 1-D Wasserstein (earth-mover) distance between the two dummy sets'
    POOLED pixel-intensity distributions (all pixels of all images in the
    set, flattened into one 1-D array per model), not a per-image or joint
    pixel-space transport plan. Implemented in closed form (no scipy
    dependency) via sorted-quantile interpolation -- see `_wasserstein_1d`.

    Deliberately does NOT reuse FarabiRobustAggregator._filter_weight_space_anomalies
    or its Virtue-weighted classifier-layer averaging. Reuses
    _resolve_aggregation_inputs, _lazy_fetch_weights, and the
    Genesis/metadata filter, exactly like the other literature baselines
    here.

    Fallback: fewer than 3 neighbors (a 2-median split needs >= 2 points per
    side to be meaningful) -> plain coordinate-wise mean over all fetched
    neighbor models, same "too small to run the real algorithm" spirit as
    Multi-Krum's n<4 fallback.

    ⚠️ COST WARNING: this is by far the most expensive baseline in this
    package -- one inversion optimization per neighbor per aggregate() call.
    See thesis-hub/docs/decisions.md (2026-09-27) for the mandatory local
    wall-clock smoke-test gate before any Kaggle campaign uses this arm.
"""

import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)

_INPUT_SHAPES = {
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


def _wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Closed-form 1-D Wasserstein (earth-mover) distance between two
    empirical distributions, via sorted-quantile interpolation. No scipy
    dependency -- see module docstring's "WASSERSTEIN APPROXIMATION" note."""
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    n = max(a.numel(), b.numel())
    qs = torch.linspace(0.0, 1.0, n)
    a_sorted, _ = torch.sort(a)
    b_sorted, _ = torch.sort(b)
    a_q = torch.quantile(a_sorted, qs)
    b_q = torch.quantile(b_sorted, qs)
    return torch.mean(torch.abs(a_q - b_q)).item()


def _two_median_split(values: List[float]) -> Tuple[List[int], List[int]]:
    """Optimal 1-D 2-median clustering: the two clusters of a 1-D point set
    that minimize each cluster's own within-cluster L1 dispersion from its
    median are always contiguous in sorted order, so trying every
    contiguous split point is exact and O(n^2) (n is <= a handful of
    neighbors here, so this is negligible cost)."""
    n = len(values)
    if n < 2:
        return list(range(n)), []
    order = sorted(range(n), key=lambda i: values[i])
    sorted_vals = [values[i] for i in order]

    def cost(group: List[float]) -> float:
        if not group:
            return 0.0
        med = sorted(group)[len(group) // 2]
        return sum(abs(v - med) for v in group)

    best_split, best_cost = 1, float("inf")
    for split in range(1, n):
        c = cost(sorted_vals[:split]) + cost(sorted_vals[split:])
        if c < best_cost:
            best_cost, best_split = c, split
    low_idx = order[:best_split]
    high_idx = order[best_split:]
    return low_idx, high_idx


class DFLDualAggregator(FarabiRobustAggregator):
    agg_type = "dfl_dual"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.alpha = float(agg_cfg.get("dfl_dual_alpha", 1.0))
        self.c1 = float(agg_cfg.get("dfl_dual_c1", 1e6))
        self.c2 = float(agg_cfg.get("dfl_dual_c2", 10.0))
        self.samples_per_class = int(agg_cfg.get("dfl_dual_samples_per_class", 10))
        self.inversion_iters = int(agg_cfg.get("dfl_dual_inversion_iters", 50))
        self.inversion_lr = float(agg_cfg.get("dfl_dual_inversion_lr", 0.1))
        self.num_classes = int(self.cfg.get("model", {}).get("num_classes", 10))
        dataset_name = str(self.cfg.get("dataset", {}).get("name", "mnist")).lower()
        self.input_shape = _INPUT_SHAPES.get(dataset_name, (1, 28, 28))
        logger.info(
            f"⚖️ [DFL-DUAL] Initialized (alpha={self.alpha}, C1={self.c1}, C2={self.c2}, "
            f"samples/class={self.samples_per_class}, inversion_iters={self.inversion_iters}) "
            f"-- OWN reconstruction of an under-specified algorithm, see module docstring."
        )

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [DFL-DUAL] Payloads list is empty. Aborting.")
            return {}

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
            logger.error("❌ [DFL-DUAL] All payloads were empty/Genesis. Aborting.")
            return {}

        neighbor_models: List[Dict[str, torch.Tensor]] = []
        authors: List[str] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                inner = payload.get("payload", payload)
                author = (
                    payload.get("author") or payload.get("proposer")
                    or inner.get("author") or inner.get("proposer") or "unknown"
                )
                neighbor_models.append(weights)
                authors.append(author)

        if not neighbor_models:
            logger.error("❌ [DFL-DUAL] No valid models fetched. Aborting.")
            return {}

        own_weights = getattr(node, "weights", None) if node is not None else None
        cache_manager = getattr(node, "cache_manager", None) if node is not None else None
        device = getattr(node, "device", "cpu") if node is not None else "cpu"

        if own_weights is None or cache_manager is None:
            logger.warning(
                "⚠️ [DFL-DUAL] Missing own_weights/cache_manager -- cannot run "
                "trust-anchored clustering. Falling back to plain coordinate-wise mean."
            )
            return self._coordinate_mean(neighbor_models)

        K = len(neighbor_models)
        if K < 3:
            logger.warning(
                f"⚠️ [DFL-DUAL] K={K} neighbors too few for a meaningful 2-median split "
                f"(need >= 3). Falling back to plain coordinate-wise mean."
            )
            return self._coordinate_mean(neighbor_models)

        layer_names = list(own_weights.keys())
        flat = [torch.cat([own_weights[n].flatten().float() for n in layer_names if n in own_weights])]
        flat += [
            torch.cat([m[n].flatten().float() for n in layer_names if n in m])
            for m in neighbor_models
        ]  # flat[0] = own, flat[1..K] = neighbors

        own_dummy = self._real_local_sample(node)
        dummies = [own_dummy]
        for m in neighbor_models:
            dummies.append(self._invert_model(m, cache_manager, device))

        P = K + 1  # population size including own model at index 0
        # Pairwise D matrix (P x P), D[a][b] = D[b][a], diagonal unused.
        D = [[0.0] * P for _ in range(P)]
        for a in range(P):
            for b in range(a + 1, P):
                e_ab = torch.norm(flat[a] - flat[b]).item()
                w_ab = _wasserstein_1d(dummies[a], dummies[b])
                d_ab = min(w_ab + self.alpha * e_ab, self.c1)
                D[a][b] = D[b][a] = d_ab

        # Trust score of every population member w.r.t. THIS node's own model (index 0).
        trust = [1.0]  # own model's trust w.r.t. itself is irrelevant, never selected as "other"
        for a in range(1, P):
            trust.append(F.cosine_similarity(flat[0].unsqueeze(0), flat[a].unsqueeze(0)).item())

        def q_for(ref_idx: int) -> float:
            others = [i for i in range(P) if i != ref_idx]
            d_to_ref = [D[ref_idx][i] for i in others]
            low_idx, high_idx = _two_median_split(d_to_ref)
            group_low = [others[i] for i in low_idx]
            group_high = [others[i] for i in high_idx]
            trust_low = sum(trust[i] for i in group_low) / max(len(group_low), 1)
            trust_high = sum(trust[i] for i in group_high) / max(len(group_high), 1)
            best_group = group_high if trust_high >= trust_low else group_low
            return sum(D[ref_idx][i] for i in best_group) or 1e-9

        q_own = q_for(0)
        r = []
        for j in range(1, P):
            q_j = q_for(j)
            r.append(min(q_j / q_own, self.c2))

        low_idx, high_idx = _two_median_split(r)
        trust_low = sum(trust[1 + i] for i in low_idx) / max(len(low_idx), 1)
        trust_high = sum(trust[1 + i] for i in high_idx) / max(len(high_idx), 1)
        survivors_pop_idx = [1 + i for i in (high_idx if trust_high >= trust_low else low_idx)]
        survivor_model_idx = [i - 1 for i in survivors_pop_idx]  # index into neighbor_models

        logger.info(
            f"✅ [DFL-DUAL] K={K} survivors={len(survivor_model_idx)} "
            f"authors_kept={[authors[i] for i in survivor_model_idx]} q_own={q_own:.4f}"
        )

        if not survivor_model_idx:
            logger.warning("⚠️ [DFL-DUAL] Stage-2 clustering kept no neighbor -- pure self-trust this round.")
            return {name: own_weights[name] for name in layer_names}

        return self._coordinate_mean([neighbor_models[i] for i in survivor_model_idx])

    def _real_local_sample(self, node: Any) -> torch.Tensor:
        """Own model (population index 0) skips inversion and uses a small
        real local-data sample instead -- see module docstring's OWN MODEL
        note. Falls back to a zero tensor (harmless: it only affects W_0b
        distances, and E_0b -- the model-domain term -- still discriminates)
        if the node exposes no usable local batch."""
        loader = getattr(node, "train_loader", None) or getattr(node, "test_loader", None)
        try:
            if loader is not None:
                for data, _ in loader:
                    n = min(self.samples_per_class * self.num_classes, data.shape[0])
                    return data[:n].detach().float()
        except Exception as e:
            logger.warning(f"⚠️ [DFL-DUAL] Could not draw a real local sample for own model: {e}")
        return torch.zeros((1,) + self.input_shape)

    def _invert_model(self, weights: Dict[str, torch.Tensor], cache_manager: Any, device: Any) -> torch.Tensor:
        """Class-representative synthesis under a frozen neighbor model --
        see module docstring's OWN MODEL / "MODEL INVERSION" SUBSTITUTION
        note for why this differs from literal DLG."""
        model = cache_manager.checkout_model(weights)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n_total = self.samples_per_class * self.num_classes
        targets = torch.arange(self.num_classes).repeat_interleave(self.samples_per_class)
        x = torch.randn((n_total,) + self.input_shape, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=self.inversion_lr)

        try:
            for _ in range(self.inversion_iters):
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, targets.to(device))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    x.clamp_(-3.0, 3.0)  # keep inputs in a plausible normalized-pixel range
        except Exception as e:
            logger.warning(f"⚠️ [DFL-DUAL] Model inversion failed ({e}); using the random init as the dummy set.")
        finally:
            cache_manager.checkin_model(model)

        return x.detach().cpu()

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
