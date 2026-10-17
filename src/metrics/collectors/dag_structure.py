from typing import List, Dict, Any
import numpy as np
from ..interface import IMetric


class DAGStructureMetric(IMetric):

    @property
    def name(self):
        return "dag_structure"

    def get_global_columns(self) -> List[str]:
        return [
            "avg_parents",
            "max_parents",
            "single_parent_ratio",
            "multi_parent_ratio"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node) -> Dict:
        return {}

    def aggregate(self, results: List[Dict], global_state=None) -> Dict[str, Any]:

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {c:0 for c in self.get_global_columns()}

        parent_counts = [
            len(v.get("parents", []))
            for v in dag.vertices.values()
        ]

        if not parent_counts:
            return {c:0 for c in self.get_global_columns()}

        avg_parents = float(np.mean(parent_counts))
        max_parents = max(parent_counts)

        single_parent = sum(1 for p in parent_counts if p == 1)
        multi_parent = sum(1 for p in parent_counts if p > 1)

        total = len(parent_counts)

        return {
            "avg_parents": avg_parents,
            "max_parents": max_parents,
            "single_parent_ratio": single_parent / total,
            "multi_parent_ratio": multi_parent / total
        }
