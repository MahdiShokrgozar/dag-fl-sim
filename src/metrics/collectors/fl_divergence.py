import logging
import numpy as np
import torch
from typing import List, Dict, Any
from ..interface import IMetric

logger = logging.getLogger(__name__)


class FLModelDivergenceMetric(IMetric):
    """
    Federated Learning Model Divergence Metric

    This metric measures the statistical divergence between local models
    across participating nodes. High divergence may indicate data heterogeneity
    or unstable training dynamics.

    The metric computes:
        - mean model norm
        - variance of model norms
        - pairwise divergence estimate

    A node exposes its current parameters one of two ways depending on
    backend: the centralized engine's nodes carry a live `node.model`
    (torch.nn.Module, per src/simulators/engines/centralized.py's
    client_update() calls); the DAG/l2_scaling backend's nodes never do --
    BaseNode (src/nodes/base_node.py) only ever holds `node.weights`, a
    plain Dict[str, torch.Tensor] (the same attribute FarabiGlobalProxAggregator
    reads as its own-state anchor). A live nn.Module only exists transiently
    inside LocalTrainer during an actual training call and is converted back
    to a weights dict before being stored on the node. Confirmed 2026-09-19
    against a real Kaggle run: `node.model` was None for every DAG-backend
    node, which is why model_norm read 0.0 even after the np.linalg.norm(model)
    bug (see below) was fixed -- that fix was correct but never reached,
    because `if model is None: return {"model_norm": 0.0}` fired first.
    This collector now falls back to `node.weights` when `.model` is absent.
    """

    _logged_first_failure = False  # class-level: log the real cause once, not every call

    @property
    def name(self):
        return "fl_model_divergence"

    def get_global_columns(self) -> List[str]:
        return [
            "model_norm_mean",
            "model_norm_variance",
            "pairwise_divergence_mean"
        ]

    def get_node_columns(self) -> List[str]:
        return ["model_norm"]

    def collect(self, node) -> Dict:
        """
        Collect model norm from each node -- from `node.model` (nn.Module)
        if present, else `node.weights` (Dict[str, Tensor]).
        """
        model = getattr(node, "model", None)
        weights = getattr(node, "weights", None)

        try:
            with torch.no_grad():
                if model is not None and hasattr(model, "parameters"):
                    # model is a torch.nn.Module -- np.linalg.norm(model)
                    # silently raised on this (caught below) and fell back
                    # to 0.0 for every node on every call originally.
                    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                elif weights:
                    flat = torch.cat([
                        w.detach().flatten() for w in weights.values()
                        if hasattr(w, "flatten")
                    ])
                else:
                    if not FLModelDivergenceMetric._logged_first_failure:
                        logger.warning(
                            "[FL-DIVERGENCE] node %s has neither a usable `.model` nor "
                            "`.weights` -- model_norm will be 0.0 for every node.",
                            getattr(node, "node_id", "?"),
                        )
                        FLModelDivergenceMetric._logged_first_failure = True
                    return {"model_norm": 0.0}
                norm = float(flat.norm().item())
        except Exception as e:
            if not FLModelDivergenceMetric._logged_first_failure:
                logger.warning(
                    "[FL-DIVERGENCE] model_norm computation failed for node %s: %s: %s "
                    "(model=%s, weights=%s)",
                    getattr(node, "node_id", "?"),
                    type(e).__name__, e,
                    type(model).__name__ if model is not None else None,
                    type(weights).__name__ if weights is not None else None,
                )
                FLModelDivergenceMetric._logged_first_failure = True
            norm = 0.0

        return {"model_norm": norm}

    def aggregate(self, rows: List[Dict], global_state=None) -> Dict[str, Any]:
        norms = [r["model_norm"] for r in rows if "model_norm" in r]

        if not norms:
            return {c: 0.0 for c in self.get_global_columns()}

        norms = np.array(norms)

        mean_norm = float(np.mean(norms))
        var_norm = float(np.var(norms))

        pairwise = []

        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                pairwise.append(abs(norms[i] - norms[j]))

        pairwise_mean = float(np.mean(pairwise)) if pairwise else 0.0

        return {
            "model_norm_mean": mean_norm,
            "model_norm_variance": var_norm,
            "pairwise_divergence_mean": pairwise_mean
        }
