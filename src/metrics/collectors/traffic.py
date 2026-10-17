# src/metrics/collectors/traffic.py

import logging
from typing import Dict, Any, List
from src.metrics.interface import IMetric

logger = logging.getLogger(__name__)

class TrafficMetric(IMetric):
    """
    Collects and aggregates unbiased network communication overhead metrics.
    Tracks L1 (Global Ledger), L2 (Gossip/Intra-cluster), and Storage (IPFS) traffic across nodes.
    Guarantees a fair comparison between baseline FL, IronForge, and Farabi.
    """
    
    @property
    def name(self) -> str:
        return "network_traffic"

    def get_global_columns(self) -> List[str]:
        return [
            "total_l1_bytes",
            "total_l2_bytes",
            "total_ipfs_bytes",
            "total_network_overhead_bytes"
        ]

    def get_node_columns(self) -> List[str]:
        return [
            "node_id",
            "l1_bytes",
            "l2_bytes",
            "ipfs_bytes",
            "total_traffic_bytes"
        ]

    def collect(self, node: Any) -> Dict[str, Any]:
        """
        Reads traffic statistics from a single node's telemetry.
        Completely idempotent and does not mutate node state.
        """
        l1_bytes = 0
        l2_bytes = 0
        ipfs_bytes = 0
        total_traffic = 0

        if hasattr(node, 'telemetry') and hasattr(node.telemetry, 'network_stats'):
            stats = node.telemetry.network_stats
            l1_bytes = stats.get("l1_bytes_sent", 0)
            l2_bytes = stats.get("l2_bytes_sent", 0)
            ipfs_bytes = stats.get("storage_bytes_up", 0)
            
        if hasattr(node, 'telemetry') and hasattr(node.telemetry, 'traffic_tracker'):
            total_traffic = node.telemetry.traffic_tracker.get("total", 0)
        else:
            # Fallback calculation if tracker total is missing
            total_traffic = l1_bytes + l2_bytes + ipfs_bytes
            
        return {
            "node_id": getattr(node, 'node_id', 'Unknown'),
            "l1_bytes": l1_bytes,
            "l2_bytes": l2_bytes,
            "ipfs_bytes": ipfs_bytes,
            "total_traffic_bytes": total_traffic
        }

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        """
        Aggregates individual node traffic into system-wide metrics.
        """
        total_l1 = sum(r.get("l1_bytes", 0) for r in results)
        total_l2 = sum(r.get("l2_bytes", 0) for r in results)
        total_ipfs = sum(r.get("ipfs_bytes", 0) for r in results)
        
        return {
            "total_l1_bytes": total_l1,
            "total_l2_bytes": total_l2,
            "total_ipfs_bytes": total_ipfs,
            "total_network_overhead_bytes": total_l1 + total_l2 + total_ipfs
        }