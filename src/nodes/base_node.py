# src/nodes/base_node.py
from __future__ import annotations
import logging
from src.utils.run_timing import timed
import sys
import copy
import numpy as np
import torch
import random
from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING

from src.utils.fast_loader import FastTensorDataLoader
from src.dag.dag import DAG

# Import Sub-components
from src.nodes.telemetry import NodeTelemetry
from src.nodes.local_trainer import LocalTrainer

# Strategy Interfaces
from src.strategies.aggregation.base import BaseAggregator
from src.strategies.l2.base import L2Strategy
from src.strategies.selection.base import TipSelector
from src.strategies.consensus.base import ConsensusStrategy
from src.strategies.batching.base import BatchingPolicy
from src.strategies.processing.base import UpdateProcessor

if TYPE_CHECKING:
    from src.simulators.engines.base_simulator import BaseSimulator
    from src.services.storage.base import StorageEngine
    from src.utils.gpu_cache import ModelCacheManager

class BaseNode:
    """
    Generic Orchestrator for a decentralized Federated Learning network participant.
    
    Acts as a Facade, delegating specific protocol tasks to its strategic components:
    - Tip Selection & Aggregation -> Delegated to Strategy Interfaces
    - Local Training -> Delegated to LocalTrainer
    - Monitoring & Traffic -> Delegated to NodeTelemetry
    - Protocol-Specific Formatting -> Delegated to UpdateProcessor
    """
    
    def __init__(
        self,
        node_id: str,
        initial_weights: Dict[str, Any],
        data_subset: Optional[torch.utils.data.Subset],
        test_set: Optional[torch.utils.data.Dataset],
        config: Dict[str, Any],
        simulator_ref: 'BaseSimulator',
        aggregator: BaseAggregator,
        tip_selector: TipSelector,
        consensus: ConsensusStrategy,
        batch_policy: BatchingPolicy,
        processor: UpdateProcessor,
        cache_manager: 'ModelCacheManager',
        l2_strategy: Optional[L2Strategy] = None,
        reputation: Optional[Any] = None,
        capacity: float = 1.0,
        device: Any = "cpu",
        storage_engine: Optional['StorageEngine'] = None,
        **kwargs
    ):
        self.node_id = node_id
        self.logger = logging.getLogger(f"Node_{node_id}")
        self.config = copy.deepcopy(config)
        self.seen_tx_hashes: Set[str] = set()
        
        # --- Sub-components ---
        self.telemetry = NodeTelemetry(self)
        self.trainer = LocalTrainer(self)
        
        # --- Hardware & Capabilities ---
        requested_device = torch.device(device) if isinstance(device, str) else device
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.config["simulation"]["gpu_cache"] = False 
        else:
            self.device = requested_device
            
        self.capacity = capacity
        self.storage = storage_engine
        self.simulator = simulator_ref
        self.cache_manager = cache_manager
        
        self.node_type = self.config.get("type", kwargs.get("node_type", "normal"))
        self.is_malicious = self.config.get("is_malicious", kwargs.get("is_malicious", False))
        self.capability_tag = kwargs.get("capability_tag", "Normal")
        self.capabilities: Dict[str, Any] = {}
        self._inject_capabilities()
        
        # --- Training Parameters ---
        self.weights = initial_weights
        self.last_block_time = -1.0 
        self.min_block_interval = self.config.get("simulation", {}).get("min_block_interval", 1.0)
        self.batch_size = self.config.get("local_training", {}).get("batch_size", 32)
        self.epochs = self.config.get("local_training", {}).get("epochs", 1)
        
        # --- Dataloaders ---
        use_gpu_cache = self.config.get("simulation", {}).get("gpu_cache", False) and self.device.type != "cpu"
        self.train_loader = FastTensorDataLoader(
            data_subset, batch_size=self.batch_size, shuffle=True, 
            device=self.device, cache_on_device=use_gpu_cache
        ) if data_subset else None

        self.test_loader = FastTensorDataLoader(
            test_set, batch_size=self.config.get("model", {}).get("test_batch_size", 1024), 
            shuffle=False, device=self.device, cache_on_device=use_gpu_cache
        ) if test_set else None
        # ++++++++++++++++++++++++++++++++++
        self.logger.info("Initializing baseline class distribution for Zero-Shot evaluation...")
        try:
            # یک مدل موقت از کش می‌گیریم
            temp_model = self.cache_manager.checkout_model(self.weights)
            baseline_acc, baseline_classes = self.trainer.evaluate_state(temp_model)
            
            self.per_class_accuracy = baseline_classes
            self.node_accuracy = baseline_acc
            self.last_accuracy = baseline_acc
            
            # بازگرداندن مدل به کش
            self.cache_manager.checkin_model(temp_model)
        except Exception as e:
            self.logger.error(f"Failed to initialize Zero-Shot state: {e}")
            
        # ++++++++++++++++++++++++++++++++++
        
        # --- Architecture Components ---
        self.local_dag = DAG()
        self.pending_updates: List[Any] = []
        
        self.aggregator = aggregator
        self.tip_selector = tip_selector
        self.consensus = consensus
        self.batch_policy = batch_policy
        self.processor = processor
        self.l2_strategy = l2_strategy
        self.reputation = reputation
        
        if hasattr(self, 'aggregator') and self.aggregator:
            self.aggregator.node = self
        if hasattr(self, 'consensus') and self.consensus:
            self.consensus.node = self
        if hasattr(self, 'tip_selector') and self.tip_selector:
            self.tip_selector.node = self
            
        if hasattr(self.l2_strategy, 'node'):
            pass
        
        # --- State Tracking ---
        # self.node_accuracy = 0.0
        # self.last_accuracy = 0.0
        # self.per_class_accuracy: Dict[int, float] = {}
        self.is_slashed = False
        self.l2_state: Dict[str, Any] = {"role": "worker", "active_session": None}
        self.l2_cluster_size = 1
        
        base_queue_cap = self.config.get("simulation", {}).get("base_queue_capacity", 100)
        self.queue_capacity = max(int(base_queue_cap * self.capacity), 2)

        self.logger.info(f"Node {self.node_id} initialized with l2_strategy: {self.l2_strategy.__class__.__name__ if self.l2_strategy else 'None'}, reputation: {self.get_current_reputation()}, capacity: {self.capacity}, queue_capacity: {self.queue_capacity}")

    def _inject_capabilities(self) -> None:
        """Assigns simulated hardware capabilities."""
        self.capabilities["tag"] = self.capability_tag
        self.capabilities["bandwidth_mbps"] = max(1.0, 10.0 * self.capacity)
    
    def calculate_upload_delay(self, size_bytes: int) -> float:
        """
        Calculates realistic network upload delay based on the node's hardware capabilities.
        Ensures fair and protocol-agnostic benchmarking across all strategies.
        """
        size_mb = size_bytes / (1024 * 1024)
        bandwidth_mbps = self.capabilities.get("bandwidth_mbps", 10.0)
        bandwidth_MBps = bandwidth_mbps / 8.0
        return size_mb / bandwidth_MBps if bandwidth_MBps > 0 else 0.0
        
    def snapshot_metrics(self) -> Dict[str, Any]:
        """Provides operational metrics to the global simulator reporting system."""
        return self.telemetry.snapshot_metrics()

    def create_and_broadcast_block(self, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Main lifecycle workflow for a federated DAG node.
        Executes: Tip Selection -> Aggregation -> Training -> Processing -> Consensus -> Broadcast
        """
        
        # `NetworkProtocol.get_reputation_module()` is documented as Optional and
        # returns None by default; only FarabiProtocol overrides it. Calling
        # through it unguarded meant every other protocol died here on its FIRST
        # block: plan 24's crash check produced 8 identical
        # "'NoneType' object has no attribute 'is_quarantined'" tracebacks per
        # seed for IronForge, which therefore proposed nothing and ran zero PoL
        # verifications. DB-FL and the generic protocol were broken the same way.
        # A protocol with no reputation module has no quarantine concept, so the
        # right behaviour is to skip the check rather than invent one - IronForge's
        # incentive layer is the balance ledger in ironforge_settlement.py, which
        # has no quarantine at all.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        if self.reputation is not None and self.reputation.is_quarantined(self.node_id):
            self.logger.warning(f"🚫 Node {self.node_id} is quarantined and cannot propose blocks.")
            return None
            
        # 1. Verification & Rate Limiting
        if self.l2_strategy and not getattr(self, '_check_l2_permission', lambda: True)(): 
            return None
        if current_time - self.last_block_time < self.min_block_interval: 
            return None

        # 2. Tip Selection
        active_dag = self.local_dag if (hasattr(self, 'local_dag') and self.local_dag.vertices) else self.simulator.dag
        
        mean_parent_accuracy = 0.0
        
        parent_hashes = []
        parent_accuracies = []
        aggregated_weights = None
        
        try:
            with timed("select"):
                selection_result = self.tip_selector.select_tips(active_dag, node=self)
            parent_hashes, selection_delay = selection_result if isinstance(selection_result, tuple) else (selection_result, 0.0)
            self.telemetry.register_compute(selection_delay)

            selected_payloads = []
            parent_accuracies = []
                
            for tx_hash in parent_hashes:
                tx = self.local_dag.vertices.get(tx_hash) if hasattr(self, 'local_dag') else None
                if tx is None:
                    tx = self.simulator.dag.vertices.get(tx_hash)
                if tx is not None:
                    selected_payloads.append(tx)
                    # Prefer the oracle-verified accuracy for this parent over its
                    # self-reported payload.accuracy. A malicious node can claim
                    # any accuracy it likes there (see spoof_metadata() in
                    # src/nodes/malicious/{poisoning,label_flipping,...}.py) and
                    # this value feeds ground_truth["parent_accuracy"] below,
                    # which IronForge's anti-lazy PoL gate uses directly
                    # (learning_dist = true_acc - parent_acc, ironforge_vrf.py).
                    # Trusting the claim let one poisoning node's inflated
                    # accuracy (e.g. 96.5% claimed vs. 0.0% real) poison the
                    # baseline for every honest descendant that picked its block
                    # as a parent, rejecting them as LAZY for a gap they never
                    # actually had. Same class of bug already fixed for the Eq.
                    # (1) aggregation weight in IronForgeAggregator._measured_accuracy
                    # (2026-09-12) but missed here. See thesis-hub/docs/decisions.md,
                    # 2026-09-15.
                    oracle_record = getattr(self.simulator, "oracle_state", {}).get(tx_hash, {})
                    p_acc = oracle_record.get(
                        "accuracy",
                        tx.get("payload", {}).get("accuracy", tx.get("accuracy", 0.0)),
                    )
                    parent_accuracies.append(p_acc)
                    
            if parent_accuracies:
                mean_parent_accuracy = sum(parent_accuracies) / len(parent_accuracies)

            # 3. Model Aggregation
            self.logger.info(f"DEBUG: Node {self.node_id} selected {len(selected_payloads)} parents for aggregation.")
            # aggregated_weights = self.aggregator.aggregate(selected_payloads)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            aggregated_weights = None
            if selected_payloads and hasattr(self, 'aggregator'):
                self.logger.info(f"🔄 [NODE {self.node_id}] Attempting to aggregate using {self.aggregator.agg_type}")
                try:
                    with timed("aggregate"):
                        aggregated_weights = self.aggregator.aggregate(selected_payloads, node=self)
                    
                    if aggregated_weights:
                        self.logger.info(f"✅ [NODE {self.node_id}] Aggregation successful. Got {len(aggregated_weights)} layers.")
                    else:
                        self.logger.warning(f"⚠️ [NODE {self.node_id}] Aggregator returned empty weights.")
                except Exception as e:
                    self.logger.error(f"❌ [NODE {self.node_id}] Aggregation failed with error: {e}", exc_info=True)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        except Exception as e:
            self.logger.error(f"Tip selection/aggregation failed: {e}")
            parent_hashes, aggregated_weights, selection_delay = [], None, 0.0
            mean_parent_accuracy = 0.0
            
        if not aggregated_weights: 
            aggregated_weights = self.weights

        # 4. Local Training & Evaluation
        with timed("train"):
            new_weights, compute_time, final_accuracy = self.trainer.train_and_evaluate(aggregated_weights)
        self.telemetry.register_compute(compute_time)
        
        self.last_accuracy = final_accuracy
        self.weights = copy.deepcopy(new_weights) 

        claimed_accuracy = getattr(self.trainer, 'claimed_accuracy', final_accuracy)
        
        if not self.per_class_accuracy:
            self.logger.warning(
                f"[NODE {self.node_id}] per_class_accuracy is empty at payload build time "
                f"(round={getattr(self, 'current_round', 'unknown')}); "
                f"this will trigger neutral M3 alignment (0.5) downstream in Tier3 consensus."
            )

        # 5. Raw Data Collection
        # Gather all neutral, generic training outputs. No protocol-specific formatting here.      
        self.logger.debug(f"🔍 [X-RAY 1 - BaseNode] Node {self.node_id} | Parents: {parent_hashes} | Accuracies: {parent_accuracies}")
        raw_training_data = {
            "weights":           new_weights,
            "base_weights":      aggregated_weights, 
            "parents":           parent_hashes,
            "parent_accuracies": parent_accuracies,
            "timestamp":         current_time,
            "author":            self.node_id,
            "accuracy":          claimed_accuracy,
            "class_accuracies":  self.per_class_accuracy or {i: 0.0 for i in range(10)},
        }
        
        # 6. Protocol-Specific Payload Formatting (Strategy Pattern)
        # Processor may modify the payload (e.g., IPFS offloading) and register extra delays
        payload_data = self.processor.process_outbound(raw_training_data, node=self)

        if "given_approval_weights" not in payload_data:
            fallback_weights = {}
            if parent_hashes:
                w = 1.0 / len(parent_hashes)
                for p in parent_hashes:
                    fallback_weights[p] = w
            payload_data["given_approval_weights"] = fallback_weights
            self.logger.warning(f"⚠️ [FALLBACK] Processor missed given_approval_weights. Auto-injected equal weights: {fallback_weights}")
            
        # 7. Calculate Protocol-Agnostic Network Delay [NEW FAIR BENCHMARKING]
        import sys
        payload_meta_size = sys.getsizeof(str(payload_data))
        payload_weights_size = sum(p.numel() * 4 for p in payload_data.get("weights", {}).values()) if "weights" in payload_data else 0
        total_payload_bytes = payload_meta_size + payload_weights_size
        
        # Calculate DAG broadcast delay physically based on finalized payload size
        dag_upload_time = self.calculate_upload_delay(total_payload_bytes)
        
        # Retrieve any side-channel delays (like IPFS uploads) registered by the processor
        side_channel_upload_time = getattr(self.telemetry, 'metric_network_delay', 0.0)
        queue_delay = len(self.pending_updates) * 0.1
        
        # Total realistic delay for this entire operation
        total_block_delay = compute_time + selection_delay + dag_upload_time + side_channel_upload_time + queue_delay
        
        # Reset telemetry network delay for the next cycle
        self.telemetry.metric_network_delay = 0.0

        if not self.consensus:
            return None
            
        self.telemetry.register_compute(0.01)
        
        # 8. Consensus Request
        final_tx = self.consensus.propose_block(payload_data, self.node_id)
        if not final_tx: 
            return None

        
        block_ready_time = current_time + total_block_delay
        final_tx.update({
            "id": f"tx_{self.node_id}_{block_ready_time:.4f}",
            "hash": f"h_{self.node_id}_{block_ready_time:.4f}",
            "timestamp": block_ready_time,
            "parents": parent_hashes,
            "accuracy": final_accuracy,
            "_total_delay": total_block_delay
        })

        # Inject Ground Truth (For simulation/oracle purposes only, not broadcasted)
        final_tx["ground_truth"] = {
            "accuracy": final_accuracy,
            "parent_accuracy": mean_parent_accuracy,
            "compute_time": compute_time,
            "model_size_mb": self.config.get("model", {}).get("estimated_size_mb", 5.0),
            "author_capability_tag": self.capabilities.get("tag", "Unknown")
        }
        
        if not hasattr(self.simulator, 'oracle_state'):
            self.simulator.oracle_state = {}
        self.simulator.oracle_state[final_tx["hash"]] = final_tx["ground_truth"]
        
        # REMOVE ground_truth from transaction before any broadcast
        broadcast_tx = {k: v for k, v in final_tx.items() if k != "ground_truth"}
        
        self.last_block_time = block_ready_time
        self.pending_updates.clear()

        # 8. Network Transmission & Telemetry Routing
        try:
            # Bandwidth simulation logic
            model_size = sum(p.numel() * 4 for p in payload_data.get("weights", {}).values()) if "weights" in payload_data else 1024
            meta_size = sys.getsizeof(str(broadcast_tx))

            routed_to_l2 = False
            if self.l2_strategy and hasattr(self.l2_strategy, 'should_route_to_l2'):
                routed_to_l2 = self.l2_strategy.should_route_to_l2(self, final_tx)
                
            if routed_to_l2:
                self.telemetry.register_traffic(model_size, channel="l2")
                self.telemetry.register_traffic(meta_size, channel="l2")
                # BUG FIX (2026-09-24, thesis-hub/docs/decisions.md same date):
                # the block used to enter the shared L2 DAG here, at creation
                # time, while its timestamp is block_ready_time (in the future).
                # Its parents left the tip pool immediately, but the block itself
                # stayed invisible to peers until timestamp + latency, so 76% of
                # FARABI's local-view queries came back empty. The L1 path below
                # already waits (broadcast delay=total_block_delay); the L2 path
                # now does the same via an L2_BLOCK_READY event.
                if hasattr(self.simulator, 'schedule_event'):
                    self.simulator.schedule_event(
                        timestamp=block_ready_time,
                        event_type="L2_BLOCK_READY",
                        node_id=self.node_id,
                        payload={"tx": final_tx},
                    )
                else:
                    self.l2_strategy.validate_l2_update(final_tx, getattr(self, 'l2_state', {}))
            else:
                self.telemetry.register_traffic(model_size, channel="ipfs")
                self.telemetry.register_traffic(meta_size, channel="l1")
                if hasattr(self.simulator, 'broadcast'):
                    self.simulator.broadcast(self.node_id, final_tx, delay=total_block_delay)

        except Exception as e:
            self.logger.warning(f"Traffic calculation error: {e}")
        
        self.logger.info(f"Block Created Successfully | Node: {self.node_id} | Accuracy: {final_accuracy:.2f}%")      
        return final_tx

    def handle_incoming_transaction(self, tx: Dict[str, Any]) -> None:
        """
        Receives and processes incoming transactions via gossip.
        Adds verified transactions to the local DAG.
        """
        tx_hash = tx.get("hash") if tx else None
        if tx_hash and tx_hash in self.seen_tx_hashes:
            return
        if tx_hash:
            self.seen_tx_hashes.add(tx_hash)
            
        if tx and self.consensus and self.consensus.validate_block(tx):
            self.telemetry.register_compute(0.005)
            payload_data = tx.get('payload', {})
            self.pending_updates.append(self.processor.process_inbound(payload_data))
            self.local_dag.add_transaction(tx)
            
    def get_current_reputation(self) -> float:
        """
        Facade method to fetch the node's current reputation score, 
        regardless of the underlying protocol implementation.
        """
        if hasattr(self, 'reputation') and hasattr(self.reputation, 'get_reputation_score'):
            return float(self.reputation.get_reputation_score(self.node_id))
            
        if hasattr(self, 'reputation_score'):
            return float(self.reputation_score)
            
        return 1.0