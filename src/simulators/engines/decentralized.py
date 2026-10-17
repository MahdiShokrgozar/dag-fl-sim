# src/simulators/engines/decentralized.py

import random
import logging
import torch
from typing import Dict, Any, List, Optional

from src.simulators.engines.base_simulator import BaseSimulator, Event
from src.nodes.factory import NodeFactory
from src.models.factory import ModelFactory
from src.data.loader import load_dataset
from src.dag.dag import DAG
from src.services.storage.factory import StorageFactory

from src.protocols.factory import ProtocolFactory
from src.utils.gpu_cache import ModelCacheManager
from src.utils.fast_loader import FastTensorDataLoader

# --- Event Constants ---
class EventType:
    """Defines the types of events handled by this simulator."""
    TRAINING_COMPLETE = "TRAINING_COMPLETE"
    MESSAGE_DELIVERY = "MESSAGE_DELIVERY"

class DecentralizedSimulator(BaseSimulator):
    """
    Decentralized Federated Learning Simulator Engine.

    This class implements the logic for a DAG-based or P2P Federated Learning system.
    It manages the global oracle DAG (for evaluation purposes), handles data loading,
    node creation, and orchestrates the event-driven simulation loop.

    Attributes:
        dag (DAG): The global reference DAG (Oracle) used for tracking consensus metrics.
        peer_cache (List[str]): A cached list of all node IDs for efficient peer sampling.
        storage (Any): The storage engine instance (e.g., Mock or IPFS).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the decentralized simulator.

        Args:
            config: Global configuration dictionary.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        super().__init__(config, **kwargs)
        
        self.dag = DAG()  # Global Oracle DAG
        self.peer_cache: List[str] = []
        self.storage = None
        self.nodes = {}

        # Global Oracle to store the ground-truth state of all generated models.
        # This prevents malicious nodes from cheating via fake metadata.
        self.oracle_state: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self):
        """
        Sets up the simulation environment.
        
        1. Initializes Storage Engine.
        2. Loads and partitions the dataset.
        3. Creates nodes and assigns data shards.
        4. Schedules the initial training events (Genesis).
        """
        self.logger.info("🏗️ Initializing Decentralized Simulation...")
        
        # 1. Initialize Storage
        self.storage = StorageFactory.create(self.config)
        
        # 2. Create Nodes & Load Data
        self._create_nodes()
        self._log_data_distribution()
        
        # 3. Schedule Initial Training (Staggered start to prevent sync-lock)
        self.logger.info("⏳ Scheduling genesis training events...")
        for node_id in self.nodes:
            # Add a small random delay so not everyone finishes exactly at t=1.0
            initial_delay = random.uniform(0, 0.5)
            self.schedule_event(
                timestamp=self.current_time + initial_delay,
                event_type=EventType.TRAINING_COMPLETE,
                node_id=node_id
            )

    def _create_nodes(self):
        """
        Internal helper to load data and instantiate nodes via Factory.
        """
        try:
            # Load Data. test_shards is each node's OWN held-out local test
            # partition (same distribution as what it trains on) - not the
            # shared global_test_set, which would give every decentralized
            # node the same global view a real local evaluation never has.
            # See thesis-hub/docs/decisions.md, 2026-09-15.
            train_shards, test_shards, global_test_set = load_dataset(self.config)

            # Create Template Model
            global_model = ModelFactory.create(self.config)

            self.logger.info("🔧 Initializing Protocol and GPU Cache...")
            self.protocol = ProtocolFactory.create(self.config)

            # Cache capacity from config or default to 5
            cache_cap = self.config.get("simulation", {}).get("gpu_cache_capacity", 5)
            device = self.config.get("simulation", {}).get("device", "cpu")
            self.cache_manager = ModelCacheManager(global_model, device=device, max_capacity=cache_cap)

            # Kept ONLY for reporting (AccuracyMetric reads this via
            # node.simulator.global_test_loader to compute a
            # global_held_out_accuracy column alongside the per-node local
            # one) - never wired into any node's own test_loader/decision
            # path. Reporting "how does this look from my own local view"
            # without also reporting "does it actually generalize globally"
            # would just be the mirror image of the bug the local-partition
            # fix corrected. See thesis-hub/docs/decisions.md, 2026-09-15.
            self.global_test_loader = FastTensorDataLoader(
                global_test_set,
                batch_size=self.config.get("model", {}).get("test_batch_size", 1024),
                shuffle=False, device=torch.device(device), cache_on_device=False
            ) if global_test_set else None

            # Create Nodes
            self.nodes = NodeFactory.create_nodes(
                config=self.config,
                model_template=global_model,
                train_dataset=train_shards,
                test_dataset=test_shards, # Per-node local test partitions
                storage_engine=self.storage,
                protocol=self.protocol,
                cache_manager=self.cache_manager
                # Pass self as simulator_ref if nodes need to call back (e.g. broadcast)
                # However, usually nodes call self.simulator.broadcast if injected
            )
            
            # Inject simulator reference into nodes
            for node in self.nodes.values():
                node.simulator = self
            
            self.peer_cache = list(self.nodes.keys())
            self.logger.info(f"✅ Created {len(self.nodes)} nodes.")
            
        except Exception as e:
            self.logger.critical(f"Failed to create nodes: {e}")
            raise e

    def _handle_event(self, event: Event):
        """
        Implements the abstract method from BaseSimulator.
        Dispatches events to specific handlers based on EventType.
        
        Args:
            event: The event to process.
        """
        if event.event_type == EventType.TRAINING_COMPLETE:
            self._handle_training_complete(event.node_id)
            
        elif event.event_type == EventType.MESSAGE_DELIVERY:
            self._handle_message_delivery(event.node_id, event.payload)
            
        else:
            self.logger.warning(f"⚠️ Unknown event type: {event.event_type}")

    def _handle_training_complete(self, node_id: str):
        """
        Handles the completion of a local training round.
        
        1. Node creates a block/transaction.
        2. Transaction is added to global DAG (Oracle).
        3. Next training round is scheduled.
        """
        node = self.nodes.get(node_id)
        if not node:
            return

        # 1. Node Logic: Create Block & Broadcast
        # (The node internally calls self.simulator.broadcast)
        transaction = node.create_and_broadcast_block(self.current_time)
        
        # 2. Update Global Oracle DAG (For visualization/metrics only)
        if transaction and "hash" in transaction:
            self.dag.add_transaction(transaction)
            
            # Extract and securely store the ground-truth data in the simulator's oracle
            # Nodes only see the main transaction body, but the simulator knows the absolute truth.
            self.oracle_state[transaction["hash"]] = transaction.get("ground_truth", {})

        # 3. Schedule Next Round
        # Compute time based on node capacity (Heterogeneity)
        
        if transaction and "_total_delay" in transaction:
            time_to_next_round = transaction["_total_delay"]
            del transaction["_total_delay"] 
        else:
            # Fallback if no block was produced (e.g. rate limited or slashed)
            base_time = self.config.get('simulation', {}).get('base_training_time_per_epoch', 1.0)
            capacity = getattr(node, 'capacity', 1.0)
            penalty = getattr(node, 'delay_multiplier', 1.0)
            time_to_next_round = (base_time * penalty) / capacity if capacity > 0 else base_time
            
        self.schedule_event(
            timestamp=self.current_time + time_to_next_round,
            event_type=EventType.TRAINING_COMPLETE,
            node_id=node_id
        )

    def broadcast(self, sender_node_id: str, transaction: Dict[str, Any], delay: float = 0.0):
        """
        Simulates network broadcast using a Gossip Protocol.
        
        Args:
            sender_node_id: The ID of the node sending the message.
            transaction: The data packet to send.
            delay: Processing time that occurred *before* the broadcast (e.g. compute time).
        """
        # Gossip Parameter (Fan-out)
        k = self.config.get("network", {}).get("gossip_k", 5)
        
        # Select Peers (Random Gossip)
        # We sample k peers, excluding self
        candidates = [pid for pid in self.peer_cache if pid != sender_node_id]
        if not candidates:
            return

        target_peers = random.sample(candidates, min(len(candidates), k))

        for peer_id in target_peers:
            network_latency = self._calculate_latency()
            
            # Event Time = Current + Processing Delay + Network Latency
            arrival_time = self.current_time + delay + network_latency
            
            # Payload wrapper
            payload = {
                'target': peer_id,
                'sender': sender_node_id,
                'data': transaction
            }
            
            self.schedule_event(
                timestamp=arrival_time,
                event_type=EventType.MESSAGE_DELIVERY,
                node_id=peer_id,
                payload=payload
            )

    def _handle_message_delivery(self, node_id: str, payload: Any) -> None:
        """
        Standard message delivery handler. 
        """
        node = self.nodes.get(node_id)
        if not node:
            return

        # 1. Deliver transaction to the node
        transaction = payload.get('data', payload)
        node.handle_incoming_transaction(transaction)

        # 2. Trigger Protocol Settlement
        # We must allow the protocol to check if it's time to finalize an epoch.
        if hasattr(self, 'protocol') and self.protocol:
            # Pass all nodes, current time, and simulator ref to the maintenance method
            if hasattr(self.protocol, 'check_and_trigger_settlement'):
                self.protocol.check_and_trigger_settlement(
                    self.nodes,
                    self.current_time,
                    simulator=self
                )
                
    def _calculate_latency(self) -> float:
        """Calculates random network latency based on config."""
        latency_cfg = self.config.get('network', {}).get('latency', {})
        min_l = latency_cfg.get('min', 0.01)
        max_l = latency_cfg.get('max', 0.1)
        return random.uniform(min_l, max_l)

    def _log_data_distribution(self):
        """Logs summary of data distribution across nodes."""
        self.logger.info("=== Data Distribution Verification ===")
        # (Placeholder for detailed logging if needed)
        pass

    def finalize(self):
        """
        Final hook to save DAG or other end-of-sim artifacts.
        """
        self.logger.info(f"🏁 Simulation finalized at {self.current_time:.2f}s")
        self.logger.info(f"📊 Global DAG Size: {len(self.dag.vertices)} blocks")
        
        # Save DAG image if configured
        try:
            self.dag.plot(save_path=self.config.get("output", {}).get("directory", "output"))
        except Exception as e:
            self.logger.warning(f"Could not plot DAG: {e}")