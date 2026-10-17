from typing import List, Dict, Any
import numpy as np
from ..interface import IMetric


class DAGThroughputMetric(IMetric):

    @property
    def name(self):
        return "dag_throughput"

    def get_global_columns(self) -> List[str]:
        return [
            "total_vertices",
            "new_vertices",
            "tx_throughput"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    def aggregate(self, results: List[Dict], global_state=None) -> Dict[str, Any]:

        dag = getattr(global_state, "dag", None)
        clock = getattr(global_state, "current_time", 0)

        if dag is None:
            return {
                "total_vertices": 0,
                "new_vertices": 0,
                "tx_throughput": 0.0
            }

        vertices = dag.vertices

        total_vertices = len(vertices)

        recent = 0
        for v in vertices.values():
            ts = v.get("timestamp", clock)
            if clock - ts < 1.0:
                recent += 1

        throughput = recent

        return {
            "total_vertices": total_vertices,
            "new_vertices": recent,
            "tx_throughput": throughput
        }
