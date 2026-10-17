"""
src/metrics/collectors/tip_selection.py

Tip Selection Security Metric Collector.
Tracks the effectiveness of tip filtering mechanisms
(quarantine, slash, finality) in the DAG-based tip selection.
"""
from typing import Dict, Any, List
from statistics import mean
from ..interface import IMetric


class TipSelectionMetric(IMetric):
    """
    Measures the efficacy of DAG tip filtering mechanisms.
    Tracks how many tips were filtered by each security layer.
    """

    @property
    def name(self) -> str:
        return "tip_selection"

    def get_global_columns(self) -> List[str]:
        return [
            "total_tips_filtered_quarantined",
            "total_tips_filtered_slashed",
            "total_tips_filtered_slash_confirmed",
            "total_no_valid_tips_events",
        ]

    def get_node_columns(self) -> List[str]:
        return [
            "tips_filtered_finalized",
            "tips_filtered_quarantined",
            "tips_filtered_slash_confirmed",
            "tips_filtered_slashed",
            "no_valid_tips_after_filter",
        ]

    def collect(self, node: Any) -> Dict[str, Any]:
        stats = getattr(node, "snapshot_metrics", lambda: {})()
        return {
            "tips_filtered_finalized": int(stats.get("tips_filtered_finalized", 0)),
            "tips_filtered_quarantined": int(stats.get("tips_filtered_quarantined", 0)),
            "tips_filtered_slash_confirmed": int(stats.get("tips_filtered_slash_confirmed", 0)),
            "tips_filtered_slashed": int(stats.get("tips_filtered_slashed", 0)),
            "no_valid_tips_after_filter": int(stats.get("no_valid_tips_after_filter", 0)),
        }

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        if not results:
            return {}
        return {
            "total_tips_filtered_quarantined": sum(r.get("tips_filtered_quarantined", 0) for r in results),
            "total_tips_filtered_slashed": sum(r.get("tips_filtered_slashed", 0) for r in results),
            "total_tips_filtered_slash_confirmed": sum(r.get("tips_filtered_slash_confirmed", 0) for r in results),
            "total_no_valid_tips_events": sum(r.get("no_valid_tips_after_filter", 0) for r in results),
        }