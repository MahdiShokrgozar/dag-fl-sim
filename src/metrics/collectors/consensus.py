from typing import List, Dict, Any
from collections import deque
import numpy as np

from ..interface import IMetric

class ConsensusGlobalMetric(IMetric):

    @property
    def name(self):
        return "ConsensusGlobal"

    # --------------------------------------------------------
    # GLOBAL COLUMNS
    # --------------------------------------------------------
    def get_global_columns(self) -> List[str]:
        return [
            "dag_vertices",
            "dag_tips",
            "dag_depth",
            "avg_tip_depth",
            "max_tip_depth",
            "avg_branching_factor",
            "max_branching_factor",
            "orphan_count",
            "orphan_ratio",
            "settlement_depth",
            "settlement_distance",
            "tx_per_round",
            "confirmation_latency_mean",
            "confirmation_latency_max",
            "consensus_confidence"
        ]

    # --------------------------------------------------------
    # NODE columns (not needed)
    # --------------------------------------------------------
    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    # --------------------------------------------------------
    # HELPERS: compute depth table for DAG
    # --------------------------------------------------------
    def _compute_depths(self, dag):
        depths = {}
        queue = deque()

        # Genesis depth = 0
        g = dag.genesis['hash']
        depths[g] = 0
        queue.append(g)

        # BFS
        while queue:
            parent = queue.popleft()
            for h, v in dag.vertices.items():
                if parent in v.get("parents", []):
                    if h not in depths:
                        depths[h] = depths[parent] + 1
                        queue.append(h)

        return depths

    # --------------------------------------------------------
    # MAIN AGGREGATION
    # --------------------------------------------------------
    def aggregate(self, rows: List[Dict], global_state=None) -> Dict[str, Any]:

        dag = getattr(global_state, 'dag', None)
        clock = getattr(global_state, 'current_time', 0)

        if dag is None or len(dag.vertices) == 0:
            return {col: 0 for col in self.get_global_columns()}

        vertices = dag.vertices
        tips = list(dag.tips)

        # ---- DEPTH TABLE ----
        depths = self._compute_depths(dag)

        # ---- DAG basic stats ----
        dag_vertices = len(vertices)
        dag_tips = len(tips)
        dag_depth = max(depths.values()) if depths else 0

        # ---- TIP depths ----
        tip_depths = [depths.get(t, 0) for t in tips]

        # ---- Branching factor ----
        branching = [len(v.get("parents", [])) for v in vertices.values()]

        # ---- Orphans ----
        orphan_count = sum(1 for v in vertices.values() if any(p not in vertices for p in v["parents"]))
        orphan_ratio = orphan_count / max(1, dag_vertices)

        # ---- Settlement depth ----
        settlement = dag.last_settlement_hash
        settlement_depth = depths.get(settlement, 0)
        settlement_distance = dag_depth - settlement_depth

        # ---- Confirmation Latency ----
        ages = []
        for h, v in vertices.items():
            ts = v.get("timestamp", clock)
            ages.append(max(0.0, clock - ts))

        conf_mean = float(np.mean(ages))
        conf_max = float(np.max(ages))

        # ---- Throughput ----
        # proxy: number of vertices added in last snapshot
        tx_per_round = len(rows)  # (rows = per-node snapshots; all nodes = 1 round tick)

        # ---- Consensus Confidence ----
        # simple normalized metric: fewer tips + smaller orphan ratio = higher confidence
        confidence = 1.0 / (1 + dag_tips + orphan_ratio * 10)

        return {
            "dag_vertices": dag_vertices,
            "dag_tips": dag_tips,
            "dag_depth": dag_depth,
            "avg_tip_depth": float(np.mean(tip_depths)) if tip_depths else 0,
            "max_tip_depth": max(tip_depths) if tip_depths else 0,
            "avg_branching_factor": float(np.mean(branching)),
            "max_branching_factor": max(branching),
            "orphan_count": orphan_count,
            "orphan_ratio": orphan_ratio,
            "settlement_depth": settlement_depth,
            "settlement_distance": settlement_distance,
            "tx_per_round": tx_per_round,
            "confirmation_latency_mean": conf_mean,
            "confirmation_latency_max": conf_max,
            "consensus_confidence": confidence
        }
