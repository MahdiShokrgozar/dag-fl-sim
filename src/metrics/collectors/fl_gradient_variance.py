import numpy as np
from typing import List, Dict, Any
from ..interface import IMetric


class FLGradientVarianceMetric(IMetric):
    """
    Federated Gradient Variance Metric

    Measures the variance of gradient magnitudes across nodes.
    High gradient variance usually indicates heterogeneous data
    distributions among participants.

    This metric is useful to diagnose slow convergence
    in federated optimization.
    """
    
    @property
    def name(self):
        return "fl_gradient_variance"

    def get_global_columns(self) -> List[str]:
        return [
            "gradient_norm_mean",
            "gradient_norm_variance"
        ]

    def get_node_columns(self) -> List[str]:
        return ["gradient_norm"]

    def collect(self, node) -> Dict:
        gradient = getattr(node, "last_gradient", None)

        if gradient is None:
            return {"gradient_norm": 0.0}

        try:
            norm = float(np.linalg.norm(gradient))
        except Exception:
            norm = 0.0

        return {"gradient_norm": norm}

    def aggregate(self, rows: List[Dict], global_state=None) -> Dict[str, Any]:

        grads = [r["gradient_norm"] for r in rows]

        if not grads:
            return {c: 0.0 for c in self.get_global_columns()}

        grads = np.array(grads)

        return {
            "gradient_norm_mean": float(np.mean(grads)),
            "gradient_norm_variance": float(np.var(grads))
        }
