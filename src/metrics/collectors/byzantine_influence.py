import numpy as np
from typing import List, Dict, Any
from ..interface import IMetric


class ByzantineInfluenceMetric(IMetric):
    """
    Byzantine Influence Metric

    Estimates the influence of dominant nodes in the DAG.
    If a small subset of nodes produces a large fraction
    of transactions, the system may be vulnerable to
    Byzantine manipulation.

    This metric uses creator distribution statistics.
    """
    
    @property
    def name(self):
        return "byzantine_influence"

    def get_global_columns(self) -> List[str]:
        return [
            "top_creator_fraction",
            "creator_entropy"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    def aggregate(self, rows: List[Dict], global_state=None):

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {c: 0.0 for c in self.get_global_columns()}

        counts = {}

        for v in dag.vertices.values():
            c = v.get("creator", "unknown")
            counts[c] = counts.get(c, 0) + 1

        vals = np.array(list(counts.values()))

        if len(vals) == 0:
            return {c: 0.0 for c in self.get_global_columns()}

        total = np.sum(vals)

        probs = vals / total

        entropy = -np.sum(probs * np.log2(probs))

        top_fraction = np.max(vals) / total

        return {
            "top_creator_fraction": float(top_fraction),
            "creator_entropy": float(entropy)
        }
