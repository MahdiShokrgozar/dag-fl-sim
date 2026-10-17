from typing import List, Dict, Any
import numpy as np
from ..interface import IMetric


class TipPressureMetric(IMetric):

    @property
    def name(self):
        return "tip_pressure"

    def get_global_columns(self) -> List[str]:
        return [
            "tips_count",
            "tips_per_tx",
            "tip_entropy"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    def aggregate(self, results: List[Dict], global_state=None):

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {c:0 for c in self.get_global_columns()}

        tips = list(dag.tips)
        vertices = dag.vertices

        tips_count = len(tips)

        tips_per_tx = tips_count / max(1,len(vertices))

        parent_usage = {}

        for v in vertices.values():
            for p in v.get("parents",[]):
                parent_usage[p] = parent_usage.get(p,0)+1

        vals = list(parent_usage.values())

        if not vals:
            entropy = 0
        else:
            probs = np.array(vals)/sum(vals)
            entropy = -np.sum(probs*np.log2(probs))

        return {
            "tips_count": tips_count,
            "tips_per_tx": tips_per_tx,
            "tip_entropy": float(entropy)
        }
