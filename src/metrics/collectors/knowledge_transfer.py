"""src/metrics/collectors/knowledge_transfer.py — Observer-side only."""
from typing import Dict, Any, List
from statistics import mean
from ..interface import IMetric

class KnowledgeTransferMetric(IMetric):
    COVERAGE_THRESHOLD = 50.0   # تعریف اندازه‌گیری (سمت Observer، نه پروتکل)
    TOTAL_CLASSES = 10

    @property
    def name(self) -> str:
        return "knowledge_transfer"

    def get_global_columns(self) -> List[str]:
        return ["mean_class_coverage", "min_class_coverage",
                "max_class_coverage", "knowledge_transfer_ratio"]

    def get_node_columns(self) -> List[str]:
        return ["class_coverage", "weak_class_count"]

    def collect(self, node: Any) -> Dict[str, Any]:
        pca = getattr(node, "per_class_accuracy", {}) or {}
        vals = [float(v) for v in pca.values()]
        return {
            "class_coverage": sum(1 for v in vals if v >= self.COVERAGE_THRESHOLD),
            "weak_class_count": sum(1 for v in vals if v <= 1.0),
        }

    def aggregate(self, results: List[Dict[str, Any]], global_state: Any = None) -> Dict[str, Any]:
        if not results:
            return {}
        covs = [float(r.get("class_coverage", 0)) for r in results]
        m = mean(covs)
        return {
            "mean_class_coverage": round(m, 2),
            "min_class_coverage": round(min(covs), 2),
            "max_class_coverage": round(max(covs), 2),
            "knowledge_transfer_ratio": round(m / self.TOTAL_CLASSES, 3),
        }