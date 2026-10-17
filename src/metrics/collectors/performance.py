"""
Performance and Resource Utilization Metric Collector.
Tracks CPU usage, memory consumption, and task queue lengths.
"""

from typing import Dict, Any, List
from statistics import mean
from ..interface import IMetric

class PerformanceMetric(IMetric):
    @property
    def name(self) -> str:
        return "performance"

    def get_global_columns(self) -> List[str]:
        return ["total_dropped_packets", "avg_queue_len", "avg_cpu_utilization"]

    def get_node_columns(self) -> List[str]:
        return ["node_cpu_virtual", "node_queue_len", "node_dropped"]

    def collect(self, node: Any) -> Dict[str, Any]:
        """Safely extracts performance data using defensive defaults."""
        stats = getattr(node, "snapshot_metrics", lambda: {})()
        
        return {
            "node_cpu_virtual": float(stats.get("node_cpu_virtual", getattr(node, "node_cpu_virtual", 0.0))),
            "node_queue_len": int(stats.get("node_queue_len", getattr(node, "node_queue_len", 0))),
            "node_dropped": int(stats.get("node_dropped", getattr(node, "node_dropped", 0)))
        }

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        if not results:
            return {}
            
        queues = [r.get("node_queue_len", 0) for r in results]
        cpus = [r.get("node_cpu_virtual", 0.0) for r in results]
        
        return {
            "total_dropped_packets": sum(r.get("node_dropped", 0) for r in results),
            "avg_queue_len": float(mean(queues)) if queues else 0.0,
            "avg_cpu_utilization": float(mean(cpus)) if cpus else 0.0
        }