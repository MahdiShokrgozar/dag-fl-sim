from typing import List, Dict, Any
from collections import deque
from ..interface import IMetric


class DAGGrowthMetric(IMetric):

    @property
    def name(self):
        return "dag_growth"

    def get_global_columns(self):
        return [
            "dag_vertices",
            "dag_depth",
            "dag_width"
        ]

    def get_node_columns(self):
        return []

    def collect(self, node):
        return {}

    def _compute_depth(self, dag):

        depths = {}
        q = deque()

        g = dag.genesis["hash"]

        depths[g] = 0
        q.append(g)

        while q:
            p = q.popleft()

            for h, v in dag.vertices.items():
                if p in v.get("parents", []):

                    if h not in depths:
                        depths[h] = depths[p] + 1
                        q.append(h)

        return depths

    def aggregate(self, results: List[Dict], global_state=None):

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {
                "dag_vertices": 0,
                "dag_depth": 0,
                "dag_width": 0
            }

        depths = self._compute_depth(dag)

        dag_depth = max(depths.values()) if depths else 0

        dag_width = len(dag.tips)

        return {
            "dag_vertices": len(dag.vertices),
            "dag_depth": dag_depth,
            "dag_width": dag_width
        }
