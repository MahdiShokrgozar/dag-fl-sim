"""
Module: lightyear_aggregator.py
Description:
    "LIGHTYEAR" (Konstantin & Mukhopadhyay, "Don't Reach for the Stars:
    Rethinking Topology for Resilient Federated Learning", arXiv:2508.05224,
    preprint) as a literature-faithful external baseline for the Q1
    decentralized-aggregation benchmark.

    Each client i scores every neighbor j's model against i's OWN local
    labeled validation set (never the neighbor's data, never a global set):

        A_ij = (A_acc + A_ECE + A_conf) / 3
        A_acc  = 1 - |Acc(h_i; V_i) - Acc(h_j; V_i)|
        A_ECE  = 1 - |ECE(h_i; V_i) - ECE(h_j; V_i)|
        A_conf = 1 - mean_x |P_i(x) - P_j(x)|,  P(x) = max(softmax(h(x)))

    Selection is a HARD threshold, not a soft weight: S_i = {theta_j :
    A_ij >= tau}. tau=0.75 is the paper's own heuristic for classification
    tasks (Appendix "Hyperparameters tau and gamma") -- picked empirically
    for their datasets, not derived from theory; carrying it over to this
    repo's 10-node/Dirichlet(0.1)/MNIST setting is an assumption, not a
    validated choice, and must be disclosed as such.

    AGGREGATION FORMULA -- RECONSTRUCTED, NOT A VERBATIM QUOTE: the paper
    states gamma is nominally "round-dependent" but its own experiments use
    a CONSTANT gamma=0.95 throughout, and states that gamma=1.0 degenerates
    "exactly to plain FedAvg over S_i". The exact equation number was not
    retained from the literature-review pass behind this file, so the
    update rule implemented here --

        theta_i^{new} = gamma * mean(S_i) + (1 - gamma) * theta_i

    -- is OUR reconstruction from that one stated degenerate case (it
    satisfies gamma=1.0 => plain FedAvg over S_i, and gamma=0 => pure
    self-trust), not a transcription of the paper's own numbered equation.
    Flag this in the paper as an interpretation, and revisit if the
    original text is re-read.

    TOPOLOGY CLAIM DOES NOT APPLY HERE: the paper's "rethinking topology"
    framing is about star-vs-P2P architecture choice, not about which
    specific peers a node gossips with -- in their P2P setting each client
    already sees ALL other clients, and the agreement-score+threshold IS
    the aggregation-rule-level selection. It does not prescribe changing
    this protocol's gossip_k/num_tips. This maps onto the aggregation-rule
    slot exactly like Multi-Krum/Bulyan/RFA/WFAgg do.

    LOCAL VALIDATION SET: this repo ALREADY carves one out for every node,
    independently of this baseline -- `node.test_loader`
    (src/nodes/base_node.py), built from each node's own held-out
    `local_test_fraction` slice of its own Dirichlet shard
    (src/data/loader.py, see the comment on `local_test_fraction` there,
    2026-09-15). It is never used for local training (LocalTrainer.
    train_and_evaluate only reads data_subset) and is a genuinely local,
    label-skew-inheriting split -- exactly the "per-client held-out
    validation set" LIGHTYEAR's method needs. No new data-pipeline split
    was required; this file only adds the Acc/ECE/confidence-agreement
    scoring on top of the existing loader. (The OTHER existing use of a
    held-out set, `global_held_out_accuracy`/`global_test_loader` in
    accuracy.py, is a SEPARATE, untouched, shared evaluation-only set --
    LIGHTYEAR never touches it, so the official held-out metric stays
    uncontaminated by this baseline's own use of node.test_loader.)

    LOCAL-VAL-SET FRAGILITY UNDER SEVERE NON-IID (our own inference, not a
    claim the paper makes): under Dirichlet(0.1), a node's local held-out
    slice can cover very few classes, which the paper never analyzes. This
    implementation guards it explicitly -- see `_min_val_samples` fallback.

    Deliberately does NOT reuse FarabiRobustAggregator._filter_weight_space_anomalies
    or its Virtue-weighted classifier-layer averaging. Reuses
    _resolve_aggregation_inputs, _lazy_fetch_weights, and the
    Genesis/metadata filter, exactly like the other literature baselines
    here. No assumed-Byzantine-count f, no n-vs-f precondition -- like
    BALANCE, a per-neighbor pairwise test that works even at n=1.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logger = logging.getLogger(__name__)


def _expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """Standard binning-based ECE over max-softmax confidence vs. correctness."""
    if confidences.numel() == 0:
        return 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.numel()
    for b in range(n_bins):
        lo, hi = bin_edges[b].item(), bin_edges[b + 1].item()
        in_bin = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum().item()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean().item()
        avg_acc = correct[in_bin].float().mean().item()
        ece += (count / n) * abs(avg_conf - avg_acc)
    return ece


class LightyearAggregator(FarabiRobustAggregator):
    agg_type = "lightyear"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        agg_cfg = self.cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.tau = float(agg_cfg.get("lightyear_tau", 0.75))
        self.gamma = float(agg_cfg.get("lightyear_gamma", 0.95))
        self.min_val_samples = int(agg_cfg.get("lightyear_min_val_samples", 5))
        logger.info(f"⚖️ [LIGHTYEAR] Initialized (tau={self.tau}, gamma={self.gamma}).")

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        payloads = self._resolve_aggregation_inputs(payloads, node)

        if not payloads:
            logger.error("❌ [LIGHTYEAR] Payloads list is empty. Aborting.")
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
            logger.error("❌ [LIGHTYEAR] All payloads were empty/Genesis. Aborting.")
            return {}

        models: List[Dict[str, torch.Tensor]] = []
        authors: List[str] = []
        for payload in valid_blocks:
            weights = self._lazy_fetch_weights(payload, node)
            if weights:
                inner = payload.get("payload", payload)
                author = (
                    payload.get("author") or payload.get("proposer")
                    or inner.get("author") or inner.get("proposer") or "unknown"
                )
                models.append(weights)
                authors.append(author)

        if not models:
            logger.error("❌ [LIGHTYEAR] No valid models fetched. Aborting.")
            return {}

        own_weights = getattr(node, "weights", None) if node is not None else None
        val_loader = getattr(node, "test_loader", None) if node is not None else None
        device = getattr(node, "device", "cpu") if node is not None else "cpu"
        cache_manager = getattr(node, "cache_manager", None) if node is not None else None

        if own_weights is None or val_loader is None or cache_manager is None:
            logger.warning(
                "⚠️ [LIGHTYEAR] Missing own_weights/test_loader/cache_manager -- "
                "cannot score agreement. Falling back to plain coordinate-wise mean."
            )
            return self._coordinate_mean(models)

        n_val = sum(len(batch[1]) for batch in val_loader) if hasattr(val_loader, "__iter__") else 0
        if n_val < self.min_val_samples:
            logger.warning(
                f"⚠️ [LIGHTYEAR] Local val set too small (n={n_val} < {self.min_val_samples}) -- "
                f"agreement scoring is unreliable at this class-sparsity. "
                f"Falling back to plain coordinate-wise mean over neighbors."
            )
            return self._coordinate_mean(models)

        try:
            own_stats = self._eval_stats(own_weights, val_loader, cache_manager, device)
        except Exception as e:
            logger.error(f"❌ [LIGHTYEAR] Failed to evaluate own model: {e}", exc_info=True)
            return self._coordinate_mean(models)

        survivors: List[Dict[str, torch.Tensor]] = []
        for author, model in zip(authors, models):
            try:
                j_stats = self._eval_stats(model, val_loader, cache_manager, device)
            except Exception as e:
                logger.warning(f"⚠️ [LIGHTYEAR] Failed to evaluate neighbor {author}: {e} -- excluding.")
                continue
            a_acc = 1.0 - abs(own_stats["acc"] - j_stats["acc"])
            a_ece = 1.0 - abs(own_stats["ece"] - j_stats["ece"])
            a_conf = 1.0 - (own_stats["conf"] - j_stats["conf"]).abs().mean().item()
            a_ij = (a_acc + a_ece + a_conf) / 3.0
            logger.info(f"🔎 [LIGHTYEAR] author={author} A_ij={a_ij:.4f} (acc={a_acc:.3f} ece={a_ece:.3f} conf={a_conf:.3f})")
            if a_ij >= self.tau:
                survivors.append(model)

        logger.info(f"✅ [LIGHTYEAR] {len(survivors)}/{len(models)} neighbors passed tau={self.tau}.")

        layer_names = list(own_weights.keys())
        if not survivors:
            logger.warning("⚠️ [LIGHTYEAR] No neighbor passed the agreement threshold -- pure self-trust this round.")
            return {name: own_weights[name] for name in layer_names}

        neighbor_mean = self._coordinate_mean(survivors)
        aggregated: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            if name not in neighbor_mean:
                aggregated[name] = own_weights[name]
                continue
            own_val = own_weights[name].float()
            neigh_val = neighbor_mean[name].float()
            original_dtype = own_weights[name].dtype
            is_float = own_weights[name].is_floating_point()
            result = self.gamma * neigh_val + (1.0 - self.gamma) * own_val
            aggregated[name] = result if is_float else result.round().to(original_dtype)
        return aggregated

    def _eval_stats(self, weights: Dict[str, torch.Tensor], val_loader: Any, cache_manager: Any, device: Any) -> Dict[str, Any]:
        """Forward-pass-only stats needed for A_ij: accuracy, ECE, and the
        per-sample max-softmax confidence vector (kept in a fixed batch
        order so two models' confidence vectors are directly comparable)."""
        model = cache_manager.checkout_model(weights)
        model.eval()
        all_conf: List[torch.Tensor] = []
        all_correct: List[torch.Tensor] = []
        try:
            with torch.inference_mode():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    logits = model(data)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    all_conf.append(conf.float().cpu())
                    all_correct.append((pred == target).cpu())
        finally:
            cache_manager.checkin_model(model)

        conf_tensor = torch.cat(all_conf) if all_conf else torch.zeros(0)
        correct_tensor = torch.cat(all_correct) if all_correct else torch.zeros(0, dtype=torch.bool)
        acc = correct_tensor.float().mean().item() if correct_tensor.numel() else 0.0
        ece = _expected_calibration_error(conf_tensor, correct_tensor)
        return {"acc": acc, "ece": ece, "conf": conf_tensor}

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
