# src/metrics/collectors/consensus_stability.py
from typing import List, Dict, Any
from collections import deque
import numpy as np

from ..interface import IMetric


class ConsensusStabilityMetric(IMetric):

    @property
    def name(self):
        return "consensus_stability"

    # ------------------------------------------------------------
    # GLOBAL COLUMNS
    # ------------------------------------------------------------
    def get_global_columns(self) -> List[str]:
        return [
            "forking_factor",
            "fork_volume",
            "tip_volatility",
            "tip_churn",
            "settlement_churn",
            "max_fork_depth",
            "avg_fork_depth",
            "orphan_pressure",
            "branching_volatility",
            "dag_settling_rate",
            "finality_stability",
            "stability_score"
        ]

    # ------------------------------------------------------------
    # NODE COLUMNS (none)
    # ------------------------------------------------------------
    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node) -> Dict:
        return {}

    # ------------------------------------------------------------
    # DEPTH MAP
    # ------------------------------------------------------------
    def _compute_depths(self, dag):
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

    # ------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------
    def aggregate(self, rows: List[Dict], global_state=None) -> Dict[str, Any]:
        dag = getattr(global_state, "dag", None)
        if dag is None or len(dag.vertices) == 0:
            out = {}
            for c in self.get_global_columns():
                out[c] = 0
            return out

        tips = list(dag.tips)
        vertices = dag.vertices
        depths = self._compute_depths(dag)

        dag_depth = max(depths.values()) if depths else 0

        # --------------------------------------------------------
        # FORKING FACTOR: ratio of tips to depth
        # --------------------------------------------------------
        forking_factor = len(tips) / max(1, dag_depth)

        # --------------------------------------------------------
        # FORK VOLUME: number of children competing per parent
        # --------------------------------------------------------
        child_count = {}
        for h, v in vertices.items():
            for p in v.get("parents", []):
                child_count[p] = child_count.get(p, 0) + 1

        fork_volume = float(np.mean(list(child_count.values()))) if child_count else 0.0

        # --------------------------------------------------------
        # TIP VOLATILITY: std dev of tip depths
        # --------------------------------------------------------
        tip_depths = [depths.get(t, 0) for t in tips]
        tip_volatility = float(np.std(tip_depths)) if tip_depths else 0.0

        # --------------------------------------------------------
        # TIP CHURN: tip_count / total_vertices
        # --------------------------------------------------------
        tip_churn = len(tips) / max(1, len(vertices))

        # --------------------------------------------------------
        # SETTLEMENT CHURN: distance of settlement from frontier
        # --------------------------------------------------------
        settlement = dag.last_settlement_hash
        settlement_depth = depths.get(settlement, 0)
        settlement_churn = (dag_depth - settlement_depth)

        # --------------------------------------------------------
        # FORK DEPTHS
        # --------------------------------------------------------
        # A fork is detected where a parent has >1 child
        fork_depths = []
        for p, c in child_count.items():
            if c > 1:
                fork_depths.append(depths.get(p, 0))

        max_fork_depth = max(fork_depths) if fork_depths else 0
        avg_fork_depth = float(np.mean(fork_depths)) if fork_depths else 0.0

        # --------------------------------------------------------
        # ORPHAN PRESSURE
        # --------------------------------------------------------
        orphan_count = sum(
            1 for v in vertices.values()
            if any(p not in vertices for p in v.get("parents", []))
        )
        orphan_pressure = orphan_count / max(1, len(vertices))

        # --------------------------------------------------------
        # BRANCHING VOLATILITY (std dev of parent count)
        # --------------------------------------------------------
        parent_counts = [len(v.get("parents", [])) for v in vertices.values()]
        branching_volatility = float(np.std(parent_counts)) if parent_counts else 0.0

        # --------------------------------------------------------
        # DAG SETTLING RATE: how fast settlement depth grows
        # --------------------------------------------------------
        dag_settling_rate = settlement_depth / max(1, len(vertices))

        # --------------------------------------------------------
        # FINALITY STABILITY (1 / (tips + orphans + fork volume))
        # --------------------------------------------------------
        finality_stability = 1.0 / (
            1 +
            len(tips) +
            orphan_pressure * 10 +
            fork_volume
        )

        # --------------------------------------------------------
        # NORMALIZED STABILITY SCORE
        # --------------------------------------------------------
        stability_score = 1.0 / (
            1 +
            forking_factor * 0.4 +
            tip_volatility * 0.1 +
            orphan_pressure * 3 +
            fork_volume * 0.2 +
            settlement_churn * 0.05
        )

        # --------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------
        return {
            "forking_factor": forking_factor,
            "fork_volume": fork_volume,
            "tip_volatility": tip_volatility,
            "tip_churn": tip_churn,
            "settlement_churn": settlement_churn,
            "max_fork_depth": max_fork_depth,
            "avg_fork_depth": avg_fork_depth,
            "orphan_pressure": orphan_pressure,
            "branching_volatility": branching_volatility,
            "dag_settling_rate": dag_settling_rate,
            "finality_stability": finality_stability,
            "stability_score": stability_score
        }
