"""
Module: trust_signals.py
Description:
    Stateless feature extractor for consensus trust-signal candidates.
    Computes a fixed set of dimensionless, scale-invariant signals from the
    validated model weights, used exclusively for OFFLINE analysis.
"""

import math
import time
from typing import Dict, Optional
import torch

Weights = Dict[str, torch.Tensor]

class TrustSignalExtractor:
    """
    Pure, side-effect-free computation of candidate trust signals.
    """

    @staticmethod
    def relative_l2(a: Weights, b: Weights) -> float:
        """Relative Frobenius distance ||a - b|| / ||b|| over shared keys."""
        if not a or not b:
            return 0.0
        num = den = 0.0
        for key in a:
            if key in b:
                num += (a[key] - b[key]).float().norm().item() ** 2
                den += b[key].float().norm().item() ** 2
        return math.sqrt(num) / max(math.sqrt(den), 1e-9)

    @staticmethod
    def cosine(a: Weights, b: Weights) -> float:
        """Cosine similarity between two weight-delta dictionaries."""
        if not a or not b:
            return 0.0
        dot = na = nb = 0.0
        for key in a:
            if key in b:
                fa = a[key].float().flatten()
                fb = b[key].float().flatten()
                dot += torch.dot(fa, fb).item()
                na += torch.dot(fa, fa).item()
                nb += torch.dot(fb, fb).item()
        return dot / max(math.sqrt(na * nb), 1e-9)

    @staticmethod
    def norm(u: Weights) -> float:
        """Global Frobenius norm of a weight-delta dictionary."""
        if not u:
            return 0.0
        return math.sqrt(sum(u[key].float().norm().item() ** 2 for key in u))

    @staticmethod
    def sign_agreement(a: Weights, b: Weights) -> float:
        """Fraction of parameters whose update signs agree."""
        if not a or not b:
            return 0.0
        agree = total = 0
        for key in a:
            if key in b:
                sa = torch.sign(a[key])
                sb = torch.sign(b[key])
                agree += int((sa == sb).sum().item())
                total += sa.numel()
        return agree / max(total, 1)

    @staticmethod
    def fingerprint_scalar(w: Weights) -> float:
        """L1 mass of the classifier head (last state-dict tensor)."""
        if not w:
            return 0.0
        keys = list(w.keys())
        if not keys:
            return 0.0
        last = w[keys[-1]]
        return float(torch.sum(torch.abs(last.float())).item())

    @staticmethod
    def calculate_sparsity(w: Weights) -> float:
        """Calculates the ratio of exact zero parameters in the model."""
        if not w:
            return 0.0
        zero_count = 0
        total_params = 0
        for tensor in w.values():
            if tensor.is_floating_point():
                zero_count += int((tensor == 0.0).sum().item())
                total_params += tensor.numel()
        return zero_count / max(total_params, 1)

    @staticmethod
    def per_layer_drift_variance(w: Weights, base: Weights) -> float:
        """Variance of per-layer relative drift (layer-wise Non-IID signal)."""
        if not w or not base:
            return 0.0
        rel = [
            (w[key] - base[key]).float().norm().item()
            / max(base[key].float().norm().item(), 1e-9)
            for key in w if key in base
        ]
        if len(rel) < 2:
            return 0.0
        mean = sum(rel) / len(rel)
        return sum((x - mean) ** 2 for x in rel) / (len(rel) - 1)

    @staticmethod
    def class_entropy(per_class: Optional[Dict[int, float]]) -> float:
        """Shannon entropy of the (normalized) per-class accuracy profile."""
        if not per_class:
            return 0.0
        masses = [max(float(v), 0.0) for v in per_class.values()]
        total = sum(masses)
        if total <= 0.0:
            return 0.0
        probs = [m / total for m in masses if m > 0.0]
        return -sum(p * math.log(p) for p in probs)

    def extract(
        self,
        weights: Weights,
        base_weights: Weights,
        prev_weights: Optional[Weights],
        accuracy: float,
        parent_accuracy: float,
        m3: float,
        per_class: Optional[Dict[int, float]],
        staleness: float,
    ) -> Dict[str, float]:
        """
        Assemble one observation dictionary with per-signal timing costs.
        """
        row: Dict[str, float] = {}

        # A genesis-parent block has no aggregated base to compare against. Both
        # base-relative signals are then UNDEFINED, and the old code silently
        # returned a real-looking number for each: relative_l2 hit its
        # `if not a or not b: return 0.0` guard and reported a perfect 0.0, while
        # c5_fp divided by max(0.0, 1e-9) and blew up to ~1e9. In the honest-only
        # baseline that was 5.5% of all rows — enough to wreck any threshold set on
        # S5, and worse, an S1 of exactly 0.0 is the lazy-attacker signature, so a
        # `S1 < tau_low` gate would have flagged every honest genesis block as lazy.
        # Emit NaN so these rows are excluded from statistics rather than faked.
        base_defined = bool(base_weights)

        if base_defined:
            t0 = time.perf_counter()
            row["c1_rel_l2"] = self.relative_l2(weights, base_weights)
            row["us_c1"] = (time.perf_counter() - t0) * 1e6

            fp_w = self.fingerprint_scalar(weights)
            fp_b = self.fingerprint_scalar(base_weights)
            row["c5_fp"] = abs(fp_w - fp_b) / fp_b if fp_b > 1e-12 else float("nan")
        else:
            row["c1_rel_l2"] = float("nan")
            row["c5_fp"] = float("nan")
            row["us_c1"] = 0.0

        if prev_weights is not None and len(prev_weights) > 0:
            u_prop = {k: weights[k] - prev_weights[k] for k in weights if k in prev_weights}
            u_ref = {k: base_weights[k] - prev_weights[k] for k in base_weights if k in prev_weights}

            t0 = time.perf_counter()
            row["c2_cos"] = self.cosine(u_prop, u_ref)
            row["us_c2"] = (time.perf_counter() - t0) * 1e6

            row["c4_sign"] = self.sign_agreement(u_prop, u_ref)
        else:
            row["c2_cos"] = row["c4_sign"] = float("nan")
            row["us_c2"] = 0.0

        row["c7_delta_acc"] = (accuracy - parent_accuracy) / 100.0
        row["c8_m3"] = m3
        row["class_entropy"] = self.class_entropy(per_class)
        row["staleness"] = staleness
        return row

    # Dropped from the live probe on 2026-09-05 to cut per-row cost. Each was an
    # extra O(P) pass and all three were already eliminated from the candidate set:
    #   c6_self_jump    - redundant with c1_rel_l2 (both distance-based)
    #   layer_drift_var - redundant with c1 (rho=0.896) and the heaviest signal
    #   sparsity_ratio  - never appeared in any report's candidate table
    # The static methods above are kept so group-1 data stays re-analysable and so
    # they can be re-enabled if a later question needs them.