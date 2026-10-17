# src/simulators/engines/l2_scaling.py

import logging
from typing import Dict, Any

from src.simulators.engines.decentralized import DecentralizedSimulator
from src.simulators.engines.base_simulator import Event
from src.protocols.factory import ProtocolFactory
from src.utils.network_utils import calculate_network_payload_size

class L2ScalingSimulator(DecentralizedSimulator):
    """
    Simulation Engine dedicated to Hierarchical (Layer-2) Federated Learning.
    
    Manages independent asynchronous event loops for:
    1. L2_TOPOLOGY_UPDATE: Dynamic Re-clustering and Task Assignment
    2. L2_SETTLEMENT: L2 Finality Evaluation and L1 Rollup Anchoring
    3. L2_GOSSIP_DELIVERY: Emulated physical network latency for P2P routing
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger("L2ScalingSimulator")
        
        self.l2_strategy = None
        
        # 1. Extract dynamic configuration intervals
        self.management_interval = config.get("simulation", {}).get("l2_management_interval", 50.0)
        
        # Dynamically extract settlement interval agnostic of the specific protocol name
        active_protocol = config.get("simulation", {}).get("protocol", "unknown")
        protocols_cfg = config.get("protocols", {})
        proto_cfg = protocols_cfg.get(active_protocol, {})
        
        
        # Fallback cascade: try to find t_target (Farabi) or settlement_interval (Ironforge)
        self.settlement_interval = float(
            proto_cfg.get("l2", {}).get("t_target", 
            proto_cfg.get("reputation", {}).get("settlement_interval", self.management_interval))
        )

    def initialize(self):
        """
        Sets up the initial network state and schedules independent L2 event loops.
        """
        super().initialize()
        
        if hasattr(self, 'nodes') and self.nodes:
            first_node = next(iter(self.nodes.values()))
            self.l2_strategy = getattr(first_node, 'l2_strategy', None)
            if self.l2_strategy:
                self.logger.info(f"✅ L2 Simulator perfectly synced with Nodes' DAG Manager.")
            else:
                self.logger.warning("⚠️ L2 Simulator could not find an active L2 Strategy inside the provided nodes!")
        else:
            self.logger.error("❌ No nodes were passed to the L2 Simulator! Settlement will fail.")
        
        self.logger.info(f"Scheduling L2 Topology Updates (interval: {self.management_interval}s)...")
        self.schedule_event(timestamp=0.1, event_type="L2_TOPOLOGY_UPDATE", node_id="system_coordinator")
        
        self.logger.info(f"Scheduling L2 Settlement Checks (interval: {self.settlement_interval}s)...")
        self.schedule_event(timestamp=self.settlement_interval, event_type="L2_SETTLEMENT", node_id="system_coordinator")

    def _handle_event(self, event: Event):
        """
        Intercepts L2-specific events and delegates them to their dedicated handlers,
        falling back to the base simulator for standard events.
        """
        if event.event_type == "L2_GOSSIP_DELIVERY":
            # 💡  Route delayed gossip block directly to the target node's inbound processor
            target_node_id = event.node_id
            block = event.payload.get("block")
            target_node = self.nodes.get(target_node_id)
            
            if target_node and hasattr(target_node, 'handle_incoming_transaction'):
                target_node.handle_incoming_transaction(block)
                
        elif event.event_type == "L2_BLOCK_READY":
            # A block reaches the shared L2 DAG only once it is ready
            # (BaseNode schedules this at block_ready_time). The L2 manager is
            # shared and binds its current node in should_route_to_l2, so rebind
            # the author before validating, as the old inline call implied.
            node = self.nodes.get(event.node_id)
            tx = (event.payload or {}).get("tx")
            l2 = getattr(node, 'l2_strategy', None) if node else None
            if l2 and tx:
                l2.node = node
                l2.validate_l2_update(tx, getattr(node, 'l2_state', {}))

        elif event.event_type == "L2_TOPOLOGY_UPDATE":
            self._handle_topology_update()
            
        elif event.event_type == "L2_SETTLEMENT":
            self._handle_settlement()
            
        else:
            super()._handle_event(event)

    # ==========================================
    #   1. TOPOLOGY EVENT LOOP
    # ==========================================
    def _handle_topology_update(self):
        """Executes dynamic clustering and sub-DAG management."""
        self.logger.debug(f"⏳ [Time: {self.current_time:.2f}] Executing L2 Topology Update...")

        if self.l2_strategy:
            # Reconstruct topology without interfering with settlement states
            self.l2_strategy.form_l2_topology(self.nodes, self.current_time)  
            stats = self.get_l2_statistics()
            self.logger.info(f"✅ Topology Updated. Active Clusters/Pools: {stats['active_sessions_count']}")

        # Reschedule the next topology update
        next_update_time = self.current_time + self.management_interval
        if next_update_time < self.max_time:
            self.schedule_event(timestamp=next_update_time, event_type="L2_TOPOLOGY_UPDATE", node_id="system_coordinator")

    # ==========================================
    #   2. SETTLEMENT EVENT LOOP
    # ==========================================
    def _handle_settlement(self):
        """
        Triggers the protocol's clearing, settlement, and rollup mechanism.
        Acts as a Gatekeeper to ensure minimum batch criteria are met before L1 Commit.
        """
        if self.l2_strategy:
            self.logger.info(f"⏳ [Time: {self.current_time:.2f}] Executing L2 Settlement...")
            # Check if conditions are met before generating the payload
            if hasattr(self.l2_strategy, 'should_commit_to_l1') and self.l2_strategy.should_commit_to_l1(current_time=int(self.current_time)):
                
                if hasattr(self.l2_strategy, 'create_commitment_payload'):
                    # nodes_ref is what lets a settlement strategy (e.g.
                    # IronForge's) sync its own ledger's slashed/rewarded state
                    # back onto the physical node objects (node.is_slashed).
                    # Omitting it doesn't crash anything - create_commitment_payload
                    # defaults to an empty dict - but it means is_slashed silently
                    # never becomes True under this backend: the tip selector
                    # (src/strategies/selection/ironforge_selection.py) never
                    # excludes a slashed node's blocks, and the security metrics
                    # (src/metrics/collectors/security.py) read a permanent 0.
                    l2_metadata = {'current_time': self.current_time, 'nodes_ref': self.nodes}
                    commitment_payload = self.l2_strategy.create_commitment_payload(None, l2_metadata)
                    
                    total_committed = commitment_payload.get('total_settled_in_batch', 0)
                    payload_size_bytes = calculate_network_payload_size(commitment_payload)
                    
                    sequencer_node = next(iter(self.nodes.values()))
                    if hasattr(sequencer_node, 'telemetry'):
                        sequencer_node.telemetry.register_traffic(payload_size_bytes, channel="l1")
                        self.logger.debug(f"[L1 METRICS] Sequencer {sequencer_node.node_id} transmitted {payload_size_bytes} bytes to Smart Contract.")
                        
                    self.logger.info(f"[⚖️ L1 COMMIT SUCCESS] Anchored a batch of {total_committed} blocks to L1 at {self.current_time:.2f}s.")
            else:
                self.logger.debug(f"[L2 SETTLEMENT] Settlement check evaluated at {self.current_time:.2f}s: Threshold criteria not yet met.")

        # Reschedule the next settlement evaluation
        next_settle_time = self.current_time + self.settlement_interval
        if next_settle_time <= self.max_time:
            self.schedule_event(timestamp=next_settle_time, event_type="L2_SETTLEMENT", node_id="system_coordinator")

    def finalize(self):
        """
        Final execution hook. Ensures the last pending epoch is evaluated and settled 
        before the simulator shuts down entirely.
        """
        self.logger.info("⚖️ [L2 SETTLEMENT] Forcing final epoch evaluation before shutdown...")
        if self.l2_strategy:
            # Force evaluate settlement on exit to capture remaining finalized blocks
            if hasattr(self.l2_strategy, 'should_commit_to_l1') and self.l2_strategy.should_commit_to_l1(current_time=int(self.current_time)):
                 if hasattr(self.l2_strategy, 'create_commitment_payload'):
                    self.l2_strategy.create_commitment_payload(None, {'current_time': self.current_time})
            
        super().finalize()

    # --- Helper Methods for Telemetry & Metrics ---
    def get_l2_statistics(self) -> Dict[str, Any]:
        """
        Gathers real-time statistics regarding L2 topology and clustering state.
        """
        if not self.l2_strategy: 
            return {}
            
        # Dynamically extract active cluster IDs and Head nodes (if applicable to the protocol)
        active_clusters = {n.l2_state.get('cluster_id') or n.l2_state.get('staked_task') for n in self.nodes.values() if getattr(n, 'l2_state', None)}
        heads = [n for n in self.nodes.values() if getattr(n, 'l2_state', {}).get('role') == 'head']
        
        return {
            "active_sessions_count": len(active_clusters),
            "total_heads": len(heads),
            "topology_version": int(self.current_time // self.management_interval)
        }