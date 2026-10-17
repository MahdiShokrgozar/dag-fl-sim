"""
Layer-2 (L2) Global Metric Collector.

This module provides the L2GlobalMetric class, which implements the IMetric
interface to collect and aggregate clustering, traffic, and stability metrics
for the Layer-2 settlement phase in hierarchical Federated Learning environments.
"""

from typing import Dict, Any, List
from statistics import mean
from collections import defaultdict

from ..interface import IMetric


class L2GlobalMetric(IMetric):
    """
    Collects and aggregates Layer-2 scaling and settlement metrics.

    Evaluates cluster dimensions, head node loads, intra/inter-cluster network
    traffic, and overall topological stability. The implementation includes
    defensive data extraction to support early simulation epochs where nodes
    may lack initialized L2 states.
    """

    @property
    def name(self) -> str:
        """Unique identifier for the metric collector."""
        return "L2GlobalMetric"

    def get_global_columns(self) -> List[str]:
        """Defines the CSV columns for epoch-level aggregated statistics."""
        return [
            "cluster_count",
            "avg_cluster_size",
            "max_cluster_size",
            "avg_head_load",
            "max_head_load",
            "intra_bytes_total",
            "inter_bytes_total",
            "intra_inter_ratio",
            "avg_round_time",
            "cluster_stability_index",
        ]

    def get_node_columns(self) -> List[str]:
        """Defines the CSV columns for node-level granular data."""
        return [
            "node_id",
            "l2_cluster_size",
            "l2_cluster_stability",
            "l2_bytes_intra",
            "l2_bytes_inter",
            "l2_head_load",
            "l2_head_queue_len",
            "l2_worker_updates",
            "l2_last_round_time",
        ]

    def collect(self, node: Any) -> Dict[str, Any]:
        """
        Safely extracts Layer-2 topology and performance metrics from a given node.

        Args:
            node (Any): The physical or virtual node instance.

        Returns:
            Dict[str, Any]: A dictionary containing the node's L2 metrics. Uninitialized
                            values default to zero or 'unknown' to prevent runtime errors.
        """
        l2_state: Dict[str, Any] = getattr(node, "l2_state", {})
        if not isinstance(l2_state, dict):
            l2_state = {}

        session_id = l2_state.get("cluster_id") or l2_state.get("active_session") or l2_state.get("staked_task") or "unknown"
        
        return {
            "session": session_id,
            "role": l2_state.get("role", "worker"),
            "l2_cluster_size": getattr(node, "l2_cluster_size", 0),
            "l2_cluster_stability": getattr(node, "l2_cluster_stability", 0.0),
            "l2_bytes_intra": getattr(node, "l2_bytes_intra", 0),
            "l2_bytes_inter": getattr(node, "l2_bytes_inter", 0),
            "l2_head_load": getattr(node, "l2_head_load", 0.0),
            "l2_head_queue_len": getattr(node, "l2_head_queue_len", 0),
            "l2_worker_updates": getattr(node, "l2_worker_updates", 0),
            "l2_last_round_time": getattr(node, "l2_last_round_time", 0.0),
        }

    def aggregate(self, rows: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        """
        Aggregates individual node metrics into network-wide Layer-2 statistics.

        Args:
            rows (List[Dict[str, Any]]): Collected metrics from all participating nodes.
            global_state (Any, optional): Global simulator state reference.

        Returns:
            Dict[str, Any]: Aggregated global metrics for the current epoch.
        """
        if not rows:
            return {}

        clusters: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for row in rows:
            session = row.get("session", "unknown")
            clusters[session].append(row)

        # Exclude uninitialized nodes from structural metrics
        valid_clusters = {k: v for k, v in clusters.items() if k != "unknown"}
        cluster_sizes = [len(nodes) for nodes in valid_clusters.values()]

        intra_total = sum(float(r.get("l2_bytes_intra", 0.0)) for r in rows)
        inter_total = sum(float(r.get("l2_bytes_inter", 0.0)) for r in rows)

        head_loads = [
            float(r.get("l2_head_load", 0.0)) 
            for r in rows if r.get("role") in ("committee", "head")
        ]
        
        round_times = [
            float(r.get("l2_last_round_time", 0.0)) 
            for r in rows if float(r.get("l2_last_round_time", 0.0)) > 0
        ]
        
        stabilities = [float(r.get("l2_cluster_stability", 0.0)) for r in rows]

        return {
            "cluster_count": len(valid_clusters),
            "avg_cluster_size": self._safe_mean(cluster_sizes),
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "avg_head_load": self._safe_mean(head_loads),
            "max_head_load": max(head_loads) if head_loads else 0.0,
            "intra_bytes_total": intra_total,
            "inter_bytes_total": inter_total,
            "intra_inter_ratio": (intra_total / max(inter_total, 1.0)),
            "avg_round_time": self._safe_mean(round_times),
            "cluster_stability_index": 1.0 / (1.0 + self._safe_mean(stabilities)),
        }

    @staticmethod
    def _safe_mean(data: List[float]) -> float:
        """Computes the mean safely, returning 0.0 for empty lists."""
        return float(mean(data)) if data else 0.0