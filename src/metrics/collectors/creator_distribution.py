from typing import List, Dict, Any
import numpy as np
from ..interface import IMetric


class CreatorDistributionMetric(IMetric):

    @property
    def name(self):
        return "creator_distribution"

    def get_global_columns(self) -> List[str]:
        return [
            "unique_creators",
            "creator_gini",
            "top_creator_share"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    def aggregate(self, results: List[Dict], global_state=None):

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {c:0 for c in self.get_global_columns()}

        counts = {}

        for v in dag.vertices.values():
            c = v.get("creator","unknown")
            counts[c] = counts.get(c,0) + 1

        values = list(counts.values())

        total = sum(values)

        if total == 0:
            return {c:0 for c in self.get_global_columns()}

        values = np.array(values)

        sorted_vals = np.sort(values)
        n = len(values)

        gini = (np.sum((2*np.arange(1,n+1)-n-1)*sorted_vals))/(n*np.sum(sorted_vals))

        top_share = max(values) / total

        return {
            "unique_creators": len(counts),
            "creator_gini": float(gini),
            "top_creator_share": top_share
        }
