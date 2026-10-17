import numpy as np
from typing import List, Dict, Any
from ..interface import IMetric


class FLAggregationDelayMetric(IMetric):
    """
    Federated Aggregation Delay Metric

    Measures the delay between local training completion
    and its inclusion in the blockchain DAG.

    This metric helps quantify the impact of network
    and consensus latency on the federated learning cycle.
    """
    @property
    def name(self):
        return "fl_aggregation_delay"

    def get_global_columns(self) -> List[str]:
        return [
            "aggregation_delay_mean",
            "aggregation_delay_max"
        ]

    def get_node_columns(self) -> List[str]:
        return ["training_finish_time"]

    def collect(self, node) -> Dict:

        finish = getattr(node, "last_training_time", None)

        return {
            "training_finish_time": finish if finish else 0.0
        }

    def aggregate(self, rows: List[Dict], global_state=None):

        clock = getattr(global_state, "current_time", 0)

        delays = []

        for r in rows:
            t = r["training_finish_time"]
            if t > 0:
                delays.append(clock - t)

        if not delays:
            return {c: 0.0 for c in self.get_global_columns()}

        delays = np.array(delays)

        return {
            "aggregation_delay_mean": float(np.mean(delays)),
            "aggregation_delay_max": float(np.max(delays))
        }
