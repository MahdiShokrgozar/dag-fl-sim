from typing import List, Dict, Any
from ..interface import IMetric


class SettlementProgressMetric(IMetric):

    @property
    def name(self):
        return "settlement_progress"

    def get_global_columns(self) -> List[str]:
        return [
            "settlement_tx_seen",
            "settlement_depth_estimate",
            "settlement_frequency"
        ]

    def get_node_columns(self) -> List[str]:
        return []

    def collect(self, node):
        return {}

    def aggregate(self, results: List[Dict], global_state=None):

        dag = getattr(global_state, "dag", None)

        if dag is None:
            return {c:0 for c in self.get_global_columns()}

        settlement_hash = dag.last_settlement_hash

        settlement_seen = 0
        settlement_depth = 0

        for v in dag.vertices.values():

            if v["hash"] == settlement_hash:
                settlement_seen = 1

            if settlement_hash in v.get("parents", []):
                settlement_depth += 1

        if not hasattr(self, "_last_seen"):
            self._last_seen = settlement_seen

        settlement_frequency = settlement_seen - self._last_seen
        self._last_seen = settlement_seen

        return {
            "settlement_tx_seen": settlement_seen,
            "settlement_depth_estimate": settlement_depth,
            "settlement_frequency": settlement_frequency
        }
