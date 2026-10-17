import numpy as np
from typing import List, Dict, Any
from ..interface import IMetric


class L2ClusterBalanceMetric(IMetric):
    """
    Layer-2 Cluster Balance Metric

    Evaluates how balanced the L2 clusters are.
    Highly imbalanced clusters may cause bottlenecks
    at cluster heads and degrade scalability.
    """
    @property
    def name(self):
        return "l2_cluster_balance"

    def get_global_columns(self) -> List[str]:
        return [
            "cluster_size_mean",
            "cluster_size_std",
            "cluster_imbalance_ratio"
        ]

    def get_node_columns(self) -> List[str]:
        return ["cluster_size"]

    def collect(self, node):

        size = getattr(node, "l2_cluster_size", 0)

        return {"cluster_size": size}

    def aggregate(self, rows: List[Dict], global_state=None):

        sizes = [r["cluster_size"] for r in rows if r["cluster_size"] > 0]

        if not sizes:
            return {c: 0.0 for c in self.get_global_columns()}

        sizes = np.array(sizes)

        mean = np.mean(sizes)
        std = np.std(sizes)

        imbalance = std / mean if mean > 0 else 0

        return {
            "cluster_size_mean": float(mean),
            "cluster_size_std": float(std),
            "cluster_imbalance_ratio": float(imbalance)
        }
