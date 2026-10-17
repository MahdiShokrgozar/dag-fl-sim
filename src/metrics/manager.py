import csv
import logging
from src.utils.run_timing import timed
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import yaml

from .collectors import (
    PerformanceMetric, 
    SecurityMetric, 
    AccuracyMetric, 
    ConsensusStabilityMetric,
    ConsensusGlobalMetric,
    CreatorDistributionMetric,
    DAGGrowthMetric,
    DAGStructureMetric,
    DAGThroughputMetric,
    FLAggregationDelayMetric,
    FLModelDivergenceMetric,
    FLGradientVarianceMetric,
    L2ClusterBalanceMetric,
    L2GlobalMetric,
    SettlementProgressMetric,
    TipPressureMetric,
    TrafficMetric,
    ByzantineInfluenceMetric,
    BackdoorASRMetric,
    TipSelectionMetric,
    KnowledgeTransferMetric
)
from .interface import IMetric

class MetricManager:
    """
    Central Orchestrator for the Simulation Metrics Subsystem.
    
    Responsibilities:
    1. Registry of Metric Collectors (Strategy Pattern).
    2. Periodic data collection from all nodes via snapshots.
    3. Aggregation of global statistics.
    4. Persistence to CSV format.
    """

    METRIC_REGISTRY = {
        "accuracy": AccuracyMetric,
        "performance": PerformanceMetric,
        "security": SecurityMetric,
        "consensus_stability": ConsensusStabilityMetric,
        "consensus_global": ConsensusGlobalMetric,
        "creator_distribution": CreatorDistributionMetric,
        "dag_growth": DAGGrowthMetric,
        "dag_structure": DAGStructureMetric,
        "dag_throughput": DAGThroughputMetric,
        "fl_aggregation_delay": FLAggregationDelayMetric,
        "fl_model_divergence": FLModelDivergenceMetric,
        "fl_gradient_variance": FLGradientVarianceMetric,
        "l2_cluster_balance": L2ClusterBalanceMetric,
        "l2_global": L2GlobalMetric,
        "settlement_progress": SettlementProgressMetric,
        "tip_pressure": TipPressureMetric,
        "network_traffic": TrafficMetric,
        "byzantine_influence": ByzantineInfluenceMetric,
        "backdoor_asr": BackdoorASRMetric,
        "tip_selection": TipSelectionMetric,
        "knowledge_transfer": KnowledgeTransferMetric
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("MetricManager")
        
        # 1. Setup Output Directory
        self.output_dir = Path(config.get("output", {}).get("directory", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Register Active Collectors
        self.collectors: List['IMetric'] = []
        
        active_preset_name = config.get("reporting", {}).get("active_metric_preset", "thesis_full")
        preset_config = config.get("metric_presets", {}).get(active_preset_name, {})
        self.logger.info(f"Loading metrics for preset: '{active_preset_name}'")
        for metric_name, is_enabled in preset_config.items():
            if is_enabled and metric_name in self.METRIC_REGISTRY:
                metric_class = self.METRIC_REGISTRY[metric_name]
                self.collectors.append(metric_class())
                self.logger.debug(f" [+] Metric Enabled: {metric_name}")
            elif is_enabled:
                self.logger.warning(f" [!] Metric '{metric_name}' is enabled in config but not found in REGISTRY!")
                
        # 3. Prepare CSV Files
        self.summary_csv = self.output_dir / "metrics_summary.csv"
        self.detailed_csv = self.output_dir / "metrics_nodes_detailed.csv"
        
        # 4. Initialize Headers
        self.global_headers = ["clock"]
        self.node_headers = ["clock", "node_id"]
        
        for collector in self.collectors:
            self.global_headers.extend(collector.get_global_columns())
            self.node_headers.extend(collector.get_node_columns())
            
        self._init_csv_files()
        
        self.simulator = None

    def attach_source(self, simulator: Any) -> None:
        """
        Links the MetricManager to the Simulator instance.
        """
        self.simulator = simulator
        self.logger.info("MetricManager successfully attached to Simulator.")

    def collect_and_save(self, timestamp: Optional[float] = None) -> None:
        """
        Compatibility method called by BaseSimulator.
        Delegates to snapshot() to perform the actual collection.
        
        Args:
            timestamp: The current simulation time (optional).
        """
        with timed("metrics"):
            self.snapshot()

    def snapshot(self, simulator: Any = None) -> None:
        """
        Takes a snapshot of the entire simulation state.
        """
        # Resolve simulator reference
        sim_ref = simulator if simulator else self.simulator
        
        if not sim_ref:
            self.logger.warning("Snapshot skipped: No simulator instance available.")
            return

        current_time = getattr(sim_ref, 'current_time', 0.0)
        
        # Handle nodes structure (Dict vs List)
        nodes_source = sim_ref.nodes
        if isinstance(nodes_source, dict):
            nodes = list(nodes_source.values())
        else:
            nodes = list(nodes_source)
        
        # 1. Collect per-node details
        node_rows = []
        for node in nodes:
            row_data = {"clock": current_time, "node_id": node.node_id}
            
            for collector in self.collectors:
                try:
                    # Collects data using node.snapshot_metrics() internally
                    metric_data = collector.collect(node)
                    if metric_data:
                        row_data.update(metric_data)
                except Exception as e:
                    # Log error but don't crash simulation
                    self.logger.error(f"Collector '{collector.name}' error on {node.node_id}: {e}")
            
            node_rows.append(row_data)

        # 2. Aggregate global stats
        global_row = {"clock": current_time}
        for collector in self.collectors:
            try:
                agg_data = collector.aggregate(node_rows, global_state=sim_ref)
                if agg_data:
                    global_row.update(agg_data)
            except Exception as e:
                self.logger.error(f"Aggregator '{collector.name}' error: {e}")

        # 3. Write to Disk
        self._write_to_csv(self.summary_csv, self.global_headers, global_row)
        self._write_to_csv(self.detailed_csv, self.node_headers, node_rows)

    def finalize(self) -> None:
        self.logger.info(f"✅ Simulation Metrics finalized in: {self.output_dir}")

    # --- Internal Helpers ---

    def _init_csv_files(self) -> None:
        for path, headers in [(self.summary_csv, self.global_headers), 
                              (self.detailed_csv, self.node_headers)]:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
            except IOError as e:
                self.logger.error(f"CSV Init Failed: {e}")

    def _write_to_csv(self, path: Path, headers: List[str], data: Union[Dict, List[Dict]]) -> None:
        try:
            with open(path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                if isinstance(data, list):
                    writer.writerows(data)
                else:
                    writer.writerow(data)
        except Exception as e:
            self.logger.error(f"Write Failed to {path.name}: {e}")