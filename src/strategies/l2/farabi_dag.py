import hashlib
import logging
from collections import deque
from platform import node
from typing import Dict, List, Any, Set, Optional, Tuple

from src.strategies.l2.base import L2Strategy

logger = logging.getLogger(__name__)

class FarabiDAGManager(L2Strategy):
    """
    Implements Layer 2 DAG Management for the FARABI architecture.
    Handles global DAG state, task-specific clustering, tip tracking, and 
    Intra-Cluster Gossip Routing without relying on a centralized head node.
    Strictly adheres to the L2Strategy interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DAG Manager with theoretical constraints.
        
        Args:
            config (Dict[str, Any]): The centralized configuration dictionary.
        """
        super().__init__(config)
        
        logger.info("=" * 50)
        logger.info("🚀 FARABI DAG MANAGER INITIALIZED 🚀")
        
        # Extract L2 theoretical constraints from the configuration
        l2_cfg = self.config.get("protocols", {}).get("farabi", {}).get("l2", {})
        
        self.w_min = l2_cfg.get("w_min", 3.0)  # Lowered default for L1 settlement
        self.theta = l2_cfg.get("theta", 0.67)
        self.t_target = l2_cfg.get("t_target", 60.0)
        # Finality rule (thesis-hub/docs/decisions.md, 2026-10-01, architecture paper).
        #   "mean_virtue" (default, every run so far): weight >= max(w_min, theta * mean
        #     virtue), which one or two approvers reach at any network size.
        #   "traffic": the design document's adaptive threshold
        #     max(w_min, theta * mean_virtue * lambda_local(t) * T), with lambda_local the
        #     rate of blocks reaching this node's view over the last finality_rate_window
        #     seconds and T = finality_target_time. Each distinct approver adds its weight
        #     once (about virtue/k on average), so this asks for a share of roughly
        #     theta * r * k * T of the pool (r = blocks per node per second), independent
        #     of N; T must stay below about 1/(theta * r * k) or no block can finalize.
        #     t_target is not reused: the L2 engine reads it as an interval fallback.
        self.finality_rule = str(l2_cfg.get("finality_rule", "mean_virtue")).lower()
        if self.finality_rule not in ("mean_virtue", "traffic"):
            raise ValueError(f"l2.finality_rule must be 'mean_virtue' or 'traffic', got {self.finality_rule!r}")
        self.finality_target_time = float(l2_cfg.get("finality_target_time", 1.0))
        self.finality_rate_window = float(l2_cfg.get("finality_rate_window", 30.0))
        self._arrival_times: deque = deque()
        self._first_arrival: Optional[float] = None
        self.min_settlement_batch = l2_cfg.get("min_settlement_batch", 2)
        self.tip_confidence_threshold = l2_cfg.get("tip_confidence_threshold", 1.0)
        # W3 (2026-09-24, thesis-hub/docs/decisions.md same date): with correct
        # L2 timing, tips only leave the pool once their approval confidence
        # reaches tip_confidence_threshold, so a node's local view held 14-18
        # tips with median age ~4s (max 10-19s) - about 3, sometimes 15,
        # rounds stale. Two opt-in alternatives, both off by default:
        #   local_tip_rule="first_visible_child": the standard Tangle rule,
        #     applied per node - a block is a tip for node i iff i can see it
        #     and cannot see any of its children.
        #   max_tip_age (seconds): drop tips older than this from the local
        #     view, unless that would leave it empty.
        self.local_tip_rule = str(l2_cfg.get("local_tip_rule", "confidence")).lower()
        max_age = l2_cfg.get("max_tip_age", None)
        self.max_tip_age = float(max_age) if max_age is not None else None
        self.children: Dict[str, List[str]] = {}
        self.staking_pools = l2_cfg.get("staking_pools", {})
        # Pool assignment (thesis-hub/docs/decisions.md, 2026-10-01): "contiguous"
        # (default, every run so far) gives each pool a contiguous slice of the sorted
        # node ids, which a node choosing its identity can exploit; "random" orders the
        # nodes by sha256(seed, epoch, node_id) before slicing, the simulator's stand-in
        # for an unpredictable per-epoch seed (a VRF or the hash of a settled block).
        # pool_epoch_seconds: reassign every that many seconds; unset = one draw per run.
        self.pool_assignment = str(l2_cfg.get("pool_assignment", "contiguous")).lower()
        if self.pool_assignment not in ("contiguous", "random"):
            raise ValueError(f"l2.pool_assignment must be 'contiguous' or 'random', got {self.pool_assignment!r}")
        _pe = l2_cfg.get("pool_epoch_seconds", None)
        self.pool_epoch_seconds = float(_pe) if _pe is not None else None
        self._pool_seed = self.config.get("simulation", {}).get("seed", 0)
        self.unsettled_blocks: Set[str] = set()
        logger.info(f"Loaded config -> w_min: {self.w_min}, theta: {self.theta}, batch_size: {self.min_settlement_batch}")
        
        if not self.staking_pools:
            logger.warning("⚠️ No staking pools defined! Defaulting to single 'Task_General' pool.")
            
        logger.info("=" * 50)
        
        # Global DAG State (The Oracle View)
        self.blocks: Dict[str, Dict[str, Any]] = {}
        self.tips: Set[str] = set()
        self.tip_accumulated_confidence: Dict[str, float] = {}
        self.approval_weights: Dict[str, float] = {}
        # block_id -> set of distinct approver node_ids that have ever
        # contributed weight to it (directly or via past-cone propagation).
        # Without this, N colluding nodes referencing each other's chain can
        # pump a block's approval_weight to finality with zero honest
        # participation - see thesis-hub/docs/decisions.md, 2026-09-03.
        self.approvers_seen: Dict[str, Set[str]] = {}
        self._approver_penalty_applied: Set[str] = set()
        self.finalized_blocks: Set[str] = set()
        self.pending_slash_evidence: Dict[str, Dict[str, Any]] = {}
        self.locally_slashed_nodes: Set[str] = set()
        
        # Dynamic reference to the local node processing the current transaction
        self.node: Optional[Any] = None

    # =========================================================================
    # Routing & L2 Network Emulation (Bypassing L1)
    # =========================================================================

    def should_route_to_l2(self, node: Any, payload: Dict[str, Any]) -> bool:
        """
        Determines if the payload should be handled exclusively by Layer 2.
        For FARABI, all operational traffic (model updates) is L2.
        """
        self.node = node  # Bind the node reference dynamically
       
        
        if node.node_id in self.locally_slashed_nodes:
            logger.warning(
                f"⛔ [SLASH-GATE] Node {node.node_id} is slashed. Block creation blocked."
            )
            return False
        
        # Quarantine gate
        if hasattr(node, 'reputation') and node.reputation.is_quarantined(node.node_id):
            logger.warning(
                f"⛔ [QUARANTINE-GATE] Node {node.node_id} is quarantined. Block creation blocked."
            )
            return False
        
        # Bind node_id to local reputation
        if hasattr(node, 'reputation') and node.reputation:
            node.reputation.node_id = node.node_id
        
        logger.debug(f"[L2 ROUTER] Node {node.node_id} is routing payload to FARABI L2 network.")
        return True

    def validate_l2_update(self, update: Dict[str, Any], l2_state: Dict[str, Any] = None) -> bool:
        """
        Validates the L2 update and acts as the Intra-Cluster Gossip Router.
        Routes the block exclusively to peers within the same task-specific L2 cluster.
        """
        author = update.get("proposer", update.get("author", "unknown"))
    
        # ✅ Slash Gate: بلاک گره slashed در L2 پذیرفته نمی‌شود
        if author in self.locally_slashed_nodes:
            logger.warning(f"⛔ [SLASH-GATE] Block from slashed node {author} rejected in L2.")
            return False
        
        # ✅ Quarantine Gate (موجود)
        if hasattr(self.node, 'reputation') and self.node.reputation.is_quarantined(author):
            logger.warning(f"⛔ [QUARANTINE-GATE] Block from quarantined node {author} rejected.")
            return False
    
        block_id = block_id = update.get("hash") or update.get("id")
        
        if not block_id:
            logger.warning(f"[L2 ROUTING] Validation failed: Payload missing block_id. Keys: {list(update.keys())}")
            return False

        proposer = update.get('proposer', 'unknown')
        if self.node and hasattr(self.node, 'reputation') and self.node.reputation.is_quarantined(proposer):
            logger.warning(f"[QUARANTINE GATE] Rejecting L2 update {block_id[:8]} from quarantined proposer {proposer}.")
            return False
    
        # 1. Store the block in the global DAG state
        self.add_block(update, simulator=getattr(self.node, 'simulator', None))

        # 2. Intra-Cluster Gossip Routing (L2 P2P Network)
        # if self.node and l2_state:
        #     my_task = l2_state.get('staked_task')
        #     simulator = getattr(self.node, 'simulator', None)
            
        #     if simulator and hasattr(simulator, 'nodes') and my_task:
        #         routed_count = 0
        #         for peer_id, peer_node in simulator.nodes.items():
        #             if peer_id != self.node.node_id:
        #                 peer_task = getattr(peer_node, 'l2_state', {}).get('staked_task')
        #                 if peer_task == my_task:
        #                     if hasattr(peer_node, 'handle_incoming_transaction'):
        #                         peer_node.handle_incoming_transaction(update)
        #                         routed_count += 1
                
        #         logger.info(f"[L2 X-RAY] Block {block_id[:8]} routed directly to {routed_count} peers in Task '{my_task}'.")
        
        return True

    # =========================================================================
    # Core DAG Mechanics
    # =========================================================================

    def add_block(self, block: Dict[str, Any], simulator: Any = None) -> None:
        """
        Unpacks the network envelope, updates approval weights, and schedules gossip latency.
        """
        block_id = block.get('hash') or block.get('id')
        author = block.get('proposer', 'unknown')
        
        if not block_id or block_id in self.blocks:
            return
        
            # ✅ Route slash evidence to dedicated handler
        from src.strategies.consensus.baseline_por_consensus import SLASH_EVIDENCE_TYPE
        if block.get("type") == SLASH_EVIDENCE_TYPE:
            self._store_slash_evidence(block)
            # Still schedule gossip so other nodes receive it
            if simulator:
                self._schedule_gossip(block, author, simulator)
            return
        
        if hasattr(self.node, 'reputation') and self.node.reputation.is_quarantined(author):
                    logger.warning(f"⛔ [QUARANTINE-GATE] Block from quarantined node {author} rejected in add_block.")
                    return        
                    
        # 1. Persist to global DAG state
        self.blocks[block_id] = block
        if self.finality_rule == "traffic":
            _now = getattr(simulator, 'current_time', None) if simulator is not None else None
            if _now is None:
                _now = block.get('timestamp', 0.0)
            self._arrival_times.append(_now)
            if self._first_arrival is None:
                self._first_arrival = _now
        self.tips.add(block_id)
        self.tip_accumulated_confidence[block_id] = 0.0
        self.approval_weights[block_id] = 1.0
        
        # 2. Extract Raw UFS Weights
        # UFS weights generated by FarabiConsensus.evaluate_and_score_tip()
        payload = block.get('payload', {})
        given_weights: Dict[str, float] = payload.get('given_approval_weights', {})
        
        # --- 🛡️ L1 Reputation Check (Sybil Resistance) ---
        # Instead of self-reporting, the DAG retrieves the author's virtue from the L1 Ledger
        voter_virtue = 1.0
        if self.node and hasattr(self.node, 'reputation'):
            voter_virtue = self.node.reputation.get_virtue_score(author)
            
        # 3. Update Tips logic & Apply Approval Weights
        parents = block.get('parents', [])
        logger.debug(f"🔍 [X-RAY 3 - DAG] Block: {block_id[:8]} | Payload given_weights: {given_weights} | Voter Virtue: {voter_virtue}")
        for p in parents:
            self.children.setdefault(p, []).append(block_id)
            raw_ufs = given_weights.get(p, 0.0)
            
            # Final Weight = Raw Evaluator Score (UFS) * Evaluator's L1 Virtue
            actual_weight = raw_ufs * voter_virtue
            
            self._propagate_weight_to_past_cone(p, actual_weight, approver_id=author)
            
            if p in self.tips:
                self.tip_accumulated_confidence[p] += actual_weight
                
                # Hard cut-off for tip removal
                if self.tip_accumulated_confidence[p] >= self.tip_confidence_threshold:
                    self.tips.remove(p)
                    # Clean up memory
                    if p in self.tip_accumulated_confidence:
                        del self.tip_accumulated_confidence[p]
                    logger.debug(f"[L2 TIPS] Tip {p[:8]} reached confidence threshold and was removed from the pool.")
                
        logger.info(f"[L2 DAG] Block {block_id[:8]} added. Total Active Tips: {len(self.tips)}")
        
        # 4. Connect to Simulator's Gossip Engine (Network Latency Simulation)
        if simulator and hasattr(simulator, 'broadcast'):
            latency = simulator._calculate_latency() if hasattr(simulator, '_calculate_latency') else 0.1
            simulator.broadcast(author, block, delay=latency)
    
    def get_local_view(self, node_id: str, current_time: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generates the localized view of the DAG for a specific node at a given time.
        Applies virtual network latency to simulate asynchronous Gossip propagation.
        
        Args:
            node_id (str): The ID of the node requesting the view.
            current_time (float): The current physical simulation time.
            
        Returns:
            Tuple: (visible_tips, visible_history)
        """
        latency_cfg = self.config.get('network', {}).get('latency', {})
        network_latency = float(latency_cfg.get('max', 1.0)) if isinstance(latency_cfg, dict) else 1.0
        
        visible_tips = []
        visible_history = []

        def _is_visible(block: Dict[str, Any]) -> bool:
            author = block.get('author', block.get('proposer', ''))
            return author == node_id or current_time >= (block.get('timestamp', 0.0) + network_latency)

        # 1. Filter currently active tips based on latency
        # sorted(): self.tips is a set, whose iteration order depends on
        # PYTHONHASHSEED - unsorted iteration here made tip visibility (and
        # therefore which models get aggregated) non-reproducible across
        # runs with an identical simulation.seed.
        if self.local_tip_rule == "first_visible_child":
            # A parent leaves self.tips when a child is inserted, but that child
            # stays invisible to others for network_latency, so parents of
            # not-yet-visible blocks are still tips from this node's side.
            candidates = set(self.tips)
            for blk in self.blocks.values():
                if not _is_visible(blk):
                    candidates.update(blk.get('parents', []))
            for tip_id in sorted(candidates):
                block = self.blocks.get(tip_id)
                if not block or not _is_visible(block):
                    continue
                kids = self.children.get(tip_id, [])
                if any(k in self.blocks and _is_visible(self.blocks[k]) for k in kids):
                    continue
                visible_tips.append(block)
        else:
            for tip_id in sorted(self.tips):
                block = self.blocks.get(tip_id)
                if not block: continue
                # Node sees its own blocks instantly; others require latency delay
                if _is_visible(block):
                    visible_tips.append(block)

        if self.max_tip_age is not None and visible_tips:
            fresh = [b for b in visible_tips if current_time - b.get('timestamp', 0.0) <= self.max_tip_age]
            if fresh:
                visible_tips = fresh

        # 2. Filter historical blocks for pool expansion (Sorted newest to oldest)
        all_blocks = sorted(list(self.blocks.values()), key=lambda b: b.get('timestamp', 0.0), reverse=True)
        for block in all_blocks:
            creation_time = block.get('timestamp', 0.0)
            author = block.get('author', block.get('proposer', ''))
            
            if creation_time == 0.0 or author == node_id or current_time >= (creation_time + network_latency):
                visible_history.append(block)
                
            if author == node_id or current_time >= (creation_time + network_latency):
                visible_history.append(block)
                
        # # Fallback to prevent deadlocks in the very first simulation epoch
        # if not visible_tips and self.tips:
        #     visible_tips = [self.blocks[t] for t in self.tips if t in self.blocks]
        #     visible_history = list(self.blocks.values())

        return visible_tips, visible_history
    
    def update_approval_weights(self, block_id: str, added_weight: float, approver_id: str = "unknown") -> bool:
        """
        Accumulates the approval weight (Virtue) for a block based on peer references.
        Triggers a Virtual Milestone if the Adaptive Finality Threshold is reached.

        Only the first approval from a given `approver_id` counts - a node
        repeatedly building on the same chain (its own blocks, or a
        colluding partner's) cannot keep re-adding weight for a block it has
        already approved once. Without this, a small colluding set could
        pump a block to finality with zero honest participation.

        Returns:
            bool: True if this approver was new and weight was actually
            added; False if this approver had already approved this block
            before (no-op) or the block doesn't exist.
        """
        if block_id not in self.blocks:
            return False

        seen = self.approvers_seen.setdefault(block_id, set())
        if approver_id in seen:
            logger.debug(
                f"⚖️ Block {block_id[:8]}: duplicate approval from {approver_id} ignored "
                f"(already counted)."
            )
            return False
        seen.add(approver_id)

        self.approval_weights[block_id] += added_weight
        current_weight = self.approval_weights[block_id]
        
        logger.debug(f"⚖️ Block {block_id[:8]} weight updated to {current_weight:.2f}")
        
        # Dynamically fetch the network's average virtue to calculate adaptive threshold
        local_network_virtue = 1.0
        if self.node and hasattr(self.node, 'reputation'):
            local_network_virtue = self.node.reputation.get_average_network_virtue()
            
        now = getattr(getattr(self.node, 'simulator', None), 'current_time', None) if self.node else None
        if self.finality_rule == "traffic":
            lam = self._local_block_rate(now if now is not None else 0.0)
            threshold = max(self.w_min, self.theta * local_network_virtue * lam * self.finality_target_time)
        else:
            threshold = max(self.w_min, self.theta * local_network_virtue)
        
        logger.debug(f"⚖️ Block {block_id[:10]} Weight: {current_weight:.2f} | Threshold: {threshold:.2f} (w_min={self.w_min}, theta={self.theta}, virtue={local_network_virtue:.1f})")
        # Organic Finality Check
        if current_weight >= threshold and block_id not in self.finalized_blocks:
            logger.info(f"🎯 Block {block_id[:10]} has reached the finality threshold with weight {current_weight:.2f}!")
            created = self.blocks[block_id].get('timestamp', None)
            if now is not None and created is not None:
                logger.info(f"[FINALITY] block={block_id} rule={self.finality_rule} latency={now - created:.3f} "
                            f"approvers={len(seen)} weight={current_weight:.2f} threshold={threshold:.2f}")
            self._trigger_virtual_milestone(block_id, current_weight, threshold)

        return True

    def get_available_tips(self) -> List[Dict[str, Any]]:
        """Fallback method. Returns raw global tips."""
        # sorted(): see determinism note in get_visible_tips() above.
        return [self.blocks[tip_id] for tip_id in sorted(self.tips)]

    def _local_block_rate(self, now: float) -> float:
        """Blocks per second reaching this node's view over the last finality_rate_window s."""
        while self._arrival_times and self._arrival_times[0] < now - self.finality_rate_window:
            self._arrival_times.popleft()
        if self._first_arrival is None:
            return 0.0
        span = min(self.finality_rate_window, max(now - self._first_arrival, 1.0))
        return len(self._arrival_times) / span

    def _trigger_virtual_milestone(self, block_id: str, weight: float, threshold: float) -> None:
        """Marks a block as settled (Organic Finality) upon threshold breach.
        Dynamically triggers L1 Settlement if the batch minimum is reached.
        """
        self.finalized_blocks.add(block_id)
        self.unsettled_blocks.add(block_id) 
        logger.info(f"[🎯 L2 MILESTONE] Block {block_id[:10]} Finalized! Weight: {weight:.2f} >= {threshold:.2f}| finalized: {block_id in self.finalized_blocks} and finalized size: {len(self.finalized_blocks)}")
        
        if self.node and hasattr(self.node, 'simulator'):
            simulator = self.node.simulator
            if len(self.unsettled_blocks) >= self.min_settlement_batch:
                # Prevent multiple triggers for the same batch
                if not getattr(self, '_settlement_triggered', False):
                    logger.info(f"[L2 TRIGGER] Batch capacity ({self.min_settlement_batch}) reached. Scheduling dynamic L1 Rollup.")
                    simulator.schedule_event(
                        timestamp=simulator.current_time + 0.01,  # Execute almost immediately
                        event_type="L2_SETTLEMENT",
                        node_id="system_coordinator"
                    )
                    self._settlement_triggered = True
                    
    # =========================================================================
    # L2 Interface Overrides (Topology & Aggregation)
    # =========================================================================
    
    def form_l2_topology(self, nodes: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """
        Creates Task-Specific Sub-DAG clusters dynamically based on L1 staking distributions.
        Supports scalable sharding for multiple nodes deterministically.
        """
        clusters: Dict[str, List[str]] = {}
        
        # Guarantee strict determinism across parallel simulation runs
        sorted_node_ids = sorted(list(nodes.keys()))
        total_nodes = len(sorted_node_ids)
        # "stake_sybil" entry reputation needs the attacker set; every node keeps its
        # own reputation view, so each one is handed the set here (first call only).
        attackers = {n for n in sorted_node_ids if getattr(nodes[n], "is_malicious", False)}
        moved = 0
        for n in sorted_node_ids:
            node_rep = getattr(nodes[n], "reputation", None)
            if node_rep is not None and hasattr(node_rep, "set_attacker_ids"):
                moved += node_rep.set_attacker_ids(attackers)
        if moved:
            logger.warning(f"⚠️ [REPUTATION SEED] {moved} ledger entries had moved before the attacker set arrived")
        if self.node and hasattr(self.node, 'reputation'):
            rep = self.node.reputation
            # Delegate to the reputation module rather than re-deriving the entry
            # value here. The previous version read self.config["reputation"], a
            # top-level key that does not exist (the real path is
            # protocols.farabi.reputation), so it silently hardcoded 50.0 and
            # ignored any configured initial_virtue.
            for node_id in sorted_node_ids:
                if node_id not in rep.ledger:
                    rep.ledger[node_id] = rep.initial_virtue_for(node_id)

            seeded = [rep.ledger[n] for n in sorted_node_ids]
            logger.info(
                f"✅ [REPUTATION SEED] Initialized {len(sorted_node_ids)} nodes "
                f"(mode={getattr(rep, 'initial_virtue_mode', 'constant')}, "
                f"virtue min={min(seeded):.1f} mean={sum(seeded)/len(seeded):.1f} "
                f"max={max(seeded):.1f})"
            )
        pool_order = self.pool_order(sorted_node_ids, current_time)
        pools = getattr(self, 'staking_pools', {})
        if not pools:
            logger.warning("⚠️ No staking pools provided. Defaulting to 'Task_General'.")
            pools = {"Task_General": 1.0}

        current_index = 0
        for pool_name, percentage in pools.items():
            pool_size = int(total_nodes * percentage)
            
            # Prevent creation of empty pools when node count is low
            if pool_size == 0 and total_nodes > current_index:
                pool_size = 1
                
            end_index = min(current_index + pool_size, total_nodes)
            assigned_nodes = pool_order[current_index:end_index]
            
            clusters[pool_name] = []
            for node_id in assigned_nodes:
                current_node = nodes[node_id]
                if not hasattr(current_node, 'l2_state') or current_node.l2_state is None:
                    current_node.l2_state = {}
                
                current_node.l2_state['staked_task'] = pool_name
                current_node.l2_state['cluster_id'] = pool_name
                current_node.l2_state['role'] = 'peer'
                current_node.l2_state['collateral_locked'] = True
                
                clusters[pool_name].append(node_id)
                
            current_index = end_index
            
        # Allocate any remainder nodes (due to float rounding) to the last pool
        if current_index < total_nodes:
            last_pool = list(pools.keys())[-1]
            for node_id in pool_order[current_index:]:
                 current_node = nodes[node_id]
                 current_node.l2_state['staked_task'] = last_pool
                 current_node.l2_state['cluster_id'] = last_pool
                 clusters[last_pool].append(node_id)
                 
        logger.info(f"✅ [L2 TOPOLOGY] Sub-DAGs configured: { {k: len(v) for k, v in clusters.items()} }")
        return {"topology": "staked_task_dags", "clusters": clusters}
    
    def pool_order(self, sorted_node_ids: List[str], current_time: float) -> List[str]:
        """Order in which nodes fill the pools (see pool_assignment in __init__)."""
        if getattr(self, 'pool_assignment', 'contiguous') != "random":
            return list(sorted_node_ids)
        epoch = 0 if self.pool_epoch_seconds is None else int(current_time // self.pool_epoch_seconds)
        key = lambda n: hashlib.sha256(f"{self._pool_seed}:{epoch}:{n}".encode()).hexdigest()
        return sorted(sorted_node_ids, key=key)

    def select_aggregator(self, cluster_id: str, nodes: List[Any]) -> Any:
        """FARABI relies on decentralized Sub-DAGs; no aggregator head is required."""
        return None

    def should_commit_to_l1(self, current_time: float) -> bool:
        """Determines if the settled L2 blocks have reached the minimum batch threshold."""
        is_ready = len(self.unsettled_blocks) >= self.min_settlement_batch
        logger.info(f"[L1 COMMIT CHECK] Unsettled Blocks: {len(self.unsettled_blocks)} | Batch Threshold: {self.min_settlement_batch} | Ready: {is_ready}")
        if is_ready:
            logger.info(f"[L1 SETTLEMENT] Batch capacity reached ({len(self.unsettled_blocks)} blocks). Triggering Rollup.")
        return is_ready

    def create_commitment_payload(self, aggregated_model: Any, l2_metadata: Dict) -> Dict[str, Any]:
        """Compiles the finalized blocks into a Merkle-root equivalent L1 payload and handles Batch Settlement."""
        logger.critical(f"📦 [L1 COMMIT PAYLOAD] Creating payload with {len(self.unsettled_blocks)} blocks")
        settled_batch = list(self.unsettled_blocks)
        epoch_rewards = {}

        # ---  L1 BATCH SETTLEMENT EXECUTION ---
        if self.node and hasattr(self.node, 'reputation'):
            for block_id in settled_batch:
                block = self.blocks.get(block_id, {})
                proposer = block.get('proposer', "unknown")
                receipt = block.get('_validation_receipt', {})
                parents = block.get('parents', [])
                payload = block.get('payload', {})
                given_weights = payload.get('given_approval_weights', {})
                
                if proposer and receipt.get("status") == "approved":
                    # پاداش خرد برای پیشنهاددهنده
                    proposer_reward_ratio = self.config.get("protocols", {}).get("farabi", {}).get("consensus", {}).get("proposer_reward_ratio", 0.05)
                    weight = receipt.get("weight", 0.0)
                    epoch_rewards[proposer] = epoch_rewards.get(proposer, 0.0) + (weight * proposer_reward_ratio)
                    
                    # پاداش اصلی برای والدهای تایید شده
                    for p_hash in parents:
                        p_block = self.blocks.get(p_hash, {})
                        p_author = p_block.get('proposer')
                        
                        if p_author and p_author not in ["system", proposer]:
                            weight_given = given_weights.get(p_hash, 0.0)
                            epoch_rewards[p_author] = epoch_rewards.get(p_author, 0.0) + weight_given

            # اعمال پاداش‌های تجمیع‌شده به Ledger لایه اول - روی لجر خصوصی هر گره،
            # نه فقط یک گره‌ی اتفاقاً bind‌شده. نگاه کنید به `_all_reputation_views`
            # و thesis-hub/docs/decisions.md، ۲۰۲۶-۰۹-۱۰.
            all_reps = self._all_reputation_views()
            current_time = getattr(self.node.simulator, 'current_time', 0.0) if self.node and hasattr(self.node, 'simulator') else 0.0
            for node_id, total_weight in epoch_rewards.items():
                if total_weight > 0:
                    for peer_rep in all_reps:
                        peer_rep.reward_approval(node_id, total_weight, current_time=current_time)
                    logger.info(f"⚖️ [L1 BATCH REWARD] Node {node_id} rewarded total weight {total_weight:.4f} across {len(all_reps)} peer ledgers")

            # A direct "majority of this proposer's blocks rejected this batch"
            # strike used to live here, calling slash_node() straight off
            # each block's single-validator `_validation_receipt` - no quorum, no
            # cross-validator corroboration at all (weaker than even the
            # pre-2026-09-09 evidence path). Every rejection reason that reaches
            # `_validation_receipt` already calls `_publish_slash_evidence` at its
            # origin in farabi_consensus.py, which peers independently re-verify
            # and vote on before quorum below can slash. Removed rather than
            # quorum-gated separately, to keep exactly one slashing path. See
            # thesis-hub/docs/decisions.md, 2026-09-10.

            # Process confirmed slash evidence blocks in L1 Settlement.
            # See `_evidence_meets_quorum` for the quorum rule and why it
            # changed 2026-09-09, and for why 2026-09-10 made it per-peer:
            # each live node independently re-evaluates quorum against its
            # OWN ledger (own total_network_stake/quorum_ratio) instead of
            # picking whichever node happens to be bound to `self.node`.
            # Peers can therefore reach different verdicts on the same
            # evidence - that is the correct per-observer behavior, not a
            # bug (see thesis-hub/docs/decisions.md, 2026-09-10).
            for evidence_id in list(self.pending_slash_evidence.keys()):
                evidence = self.pending_slash_evidence.get(evidence_id)
                if not evidence:
                    continue

                target = evidence.get("target")
                reason = evidence.get("reason", "consensus_slash")
                confirming_reps = [
                    peer_rep for peer_rep in all_reps
                    if self._evidence_meets_quorum(evidence, reputation=peer_rep)
                ]
                if confirming_reps:
                    for peer_rep in confirming_reps:
                        peer_rep.slash_node(target, reason)
                    # Approver penalty applied here too - same settlement
                    # moment, same reputation-owned call, not the immediate
                    # _check_slash_confirmation path (see its docstring).
                    # Applied to the same peers that actually confirmed
                    # quorum, not to every peer regardless of their verdict.
                    self._penalize_approvers_of_slashed_node(target, reason, confirming_reps)
                    logger.critical(
                        f"☠️ [L1 SLASH] Node {target} slashed in settlement for '{reason}' "
                        f"({len(confirming_reps)}/{len(all_reps)} peer ledgers independently confirmed quorum)."
                    )
                    del self.pending_slash_evidence[evidence_id]
                              
        payload = {
            "protocol": "FARABI_L2_ROLLUP",
            "settled_blocks_batch": settled_batch,
            "timestamp": l2_metadata.get('current_time', 0.0),
            "total_settled_in_batch": len(settled_batch)
        }
        
        logger.info(f"[⚖️ L1 COMMIT] Committing & Settling batch of {len(settled_batch)} verified blocks to L1.")
        self.unsettled_blocks.clear()
        self._settlement_triggered = False
        return payload
    
    def _propagate_weight_to_past_cone(self, start_parent_id: str, added_weight: float, approver_id: str = "unknown") -> None:
        """
        Propagates approval weight to the entire past cone (all ancestors) of
        `start_parent_id`, using an iterative BFS traversal to avoid Python's
        recursion depth limits on long-running DAGs.

        A `visited` set guards against double-counting within this single
        call when multiple branches converge on a shared ancestor (diamond
        pattern). Across separate calls (this node approving the same
        ancestor again later, e.g. by repeatedly building on a colluding
        partner's chain), `update_approval_weights` enforces the real guard:
        one approver contributes weight to a given block at most once, ever.
        If `approver_id` already approved `current_id` in a past call, that
        approval (and everything in its past cone) was already fully
        propagated then, so traversal stops there instead of re-walking
        already-counted ancestors.
        """
        visited: Set[str] = set()
        queue: List[str] = [start_parent_id]

        while queue:
            current_id = queue.pop(0)

            if current_id in visited or current_id not in self.blocks:
                continue
            visited.add(current_id)

            counted = self.update_approval_weights(current_id, added_weight, approver_id=approver_id)
            if not counted:
                continue

            queue.extend(self.blocks[current_id].get('parents', []))

    
    def add_slash_evidence(self, evidence_block: Dict[str, Any]) -> None:
        """
        Adds a SlashEvidenceBlock to the pending list.
        Called by FarabiConsensus when an anomaly is detected.
        """
        block_id = evidence_block.get("hash")
        if not block_id or block_id in self.pending_slash_evidence:
            return
        
        self.pending_slash_evidence[block_id] = evidence_block
        logger.info(
            f"🗳️ [SLASH_EVIDENCE] Added evidence {block_id[:8]} "
            f"against {evidence_block.get('target')} "
            f"(reason: {evidence_block.get('reason')})"
        )
        
        # Immediately check if this evidence is already confirmed
        self._check_slash_confirmation(block_id)


    
    def _store_slash_evidence(self, evidence_block: Dict[str, Any]) -> None:
        """Stores a slash evidence block in the pending list."""
        evidence_id = evidence_block.get("hash")
        if evidence_id and evidence_id not in self.pending_slash_evidence:
            self.pending_slash_evidence[evidence_id] = evidence_block
            logger.info(
                f"🗳️ [SLASH_EVIDENCE] Stored evidence {evidence_id[:8]} "
                f"against {evidence_block.get('target')}"
            )


    def approve_slash_evidence(self, evidence_id: str, voter_id: str, voter_virtue: float) -> None:
        """Casts an approval vote on a pending slash evidence block."""
        evidence = self.pending_slash_evidence.get(evidence_id)
        if not evidence:
            return

        approvals = evidence.get("approvals", set())
        if voter_id in approvals:
            return  # Already voted

        approvals.add(voter_id)
        evidence["approvals"] = approvals
        evidence["approval_weight"] = evidence.get("approval_weight", 0.0) + voter_virtue

        self._check_slash_confirmation(evidence_id)


    def _evidence_meets_quorum(self, evidence: Dict[str, Any], reputation: Optional[Any] = None) -> bool:
        """
        Shared quorum rule for slash evidence: weight-majority of total
        network stake AND a minimum count of distinct approvers.

        Used identically by `_check_slash_confirmation` (LOCAL_BLOCK, a
        non-persistent local stopgap) and by the L1 settlement loop in
        `create_commitment_payload` (the only place that actually mutates
        the reputation ledger - permanent, irreversible). Before 2026-09-09
        these two used DIFFERENT thresholds: settlement checked only
        `max(w_min, theta * local_network_virtue)` (~30-35 with typical
        virtue~50 nodes) with NO distinct-voter floor at all - roughly 9x
        weaker than this one, and satisfiable by a single high-virtue
        approval alone. Since settlement is the path with permanent
        consequences, having it be the WEAKER of the two had the risk
        asymmetry exactly backwards (the irreversible path should be the
        stricter one, not the reversible one - see the S1 gate's own
        cold-start design principle for the same logic). A real Kaggle run
        (group 3, mixed_attack, 2026-09-09) confirmed the consequence:
        permanent false-positive slashes against honest nodes (2/8 under
        `unified`, 8/8 under `dual`). See thesis-hub/docs/decisions.md,
        2026-09-09.

        `reputation`, when given, evaluates the threshold from THAT specific
        node's own ledger rather than whichever node happens to be bound to
        `self.node`. Reputation is deliberately a per-observer ledger (every
        node keeps its own, see `SecurityMetric._peer_votes`), so settlement
        must let each live node reach its own independent verdict from its
        own stake/quorum view instead of picking one node's ledger to stand
        for the whole network - see thesis-hub/docs/decisions.md, 2026-09-10.
        Defaults to `self.node.reputation` for `_check_slash_confirmation`,
        which only ever touches the already-global `locally_slashed_nodes`
        set, not a per-node ledger.

        `w_min`/`theta` remain as the fallback only for when no reputation
        module is available; `total_stake`/`quorum_ratio` are themselves live
        (ledger-derived) values, which carries a separate, not-yet-addressed
        drift risk (an attacked network's falling average virtue could
        weaken this same threshold over time) - tracked as an open item, not
        fixed here.
        """
        current_weight = evidence.get("approval_weight", 0.0)
        approvals = evidence.get("approvals", set())

        threshold = self.w_min
        min_voters = 3
        rep = reputation if reputation is not None else getattr(self.node, "reputation", None)
        if rep is not None:
            threshold = rep.quorum_ratio * rep.get_total_network_stake()
            min_voters = rep.slash_threshold   # کف: حداقل ۳ رأی متمایز

        return current_weight >= threshold and len(approvals) >= min_voters

    def _all_reputation_views(self) -> List[Any]:
        """
        Every live node's own reputation ledger.

        Reputation is per-observer by design (see `_evidence_meets_quorum`
        docstring and `SecurityMetric._peer_votes`), so settlement steps that
        mutate the ledger (slash, reward, approver penalty) must apply to
        every node's own view, not to whichever one node happens to be bound
        to `self.node`. Falls back to just `self.node.reputation` when no
        simulator/node registry is reachable (e.g. isolated tests) - see
        thesis-hub/docs/decisions.md, 2026-09-10.
        """
        simulator = getattr(self.node, "simulator", None) if self.node else None
        peers = getattr(simulator, "nodes", None)
        if peers:
            return [p.reputation for p in peers.values() if getattr(p, "reputation", None)]
        if self.node and getattr(self.node, "reputation", None):
            return [self.node.reputation]
        return []

    def _check_slash_confirmation(self, evidence_id: str) -> None:
        """
        Local-only, immediate defensive gate - NOT a reputation change.

        On quorum, this only updates `locally_slashed_nodes` (this node's
        own real-time policy to stop accepting new blocks from `target` -
        analogous to a local blacklist, not a globally-binding ledger
        change). It deliberately does NOT touch the reputation ledger and
        does NOT delete the evidence: only `reputation`, and only at L1
        settlement (`create_commitment_payload`, which already processes
        `pending_slash_evidence` independently), is allowed to mutate
        virtue/strikes. Previously this function called
        `rep.apply_consensus_slash()` directly, immediately, outside
        settlement - fixed per thesis-hub/docs/decisions.md, 2026-09-03
        (student: "کسی غیر از reputation و در تسویه‌حساب لایه ۱ نباید
        بتونه جریمه اعمال کنه").

        Per-observer quorum (2026-09-12): this used to call
        `_evidence_meets_quorum(evidence)` with no `reputation` argument, so
        the threshold came from `self.node.reputation` - and because
        `FarabiDAGManager` is a singleton, `self.node` is whichever ONE node
        it happens to be bound to. That is exactly the defect fixed for the
        settlement loop on 2026-09-10, which was never applied to this path.
        Consequence measured in plan 23 group 3 / seed_43: under `dual`,
        settlement confirmed quorum 36 times (`L1 SLASH`) while this gate
        fired ZERO times, so `locally_slashed_nodes` stayed empty for the
        whole 400s run - and since this set is what `_apply_security_filters`
        and the L2 `SLASH-GATE` actually consult, a confirmed poisoner kept
        publishing (92 blocks) and kept being picked as the Farabi-Prox
        anchor (522 times, 509 of them after it was already quarantined).
        Under `unified` the bound node's own view happened to reach quorum,
        so the same bug was invisible (7 LOCAL_BLOCKs). Now every live node's
        ledger is polled and ANY node reaching quorum blocks the target -
        identical semantics to the settlement loop's `if confirming_reps:`.
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        evidence = self.pending_slash_evidence.get(evidence_id)
        if not evidence:
            return

        target = evidence.get("target")

        if target in self.locally_slashed_nodes:
            return

        all_reps = self._all_reputation_views()
        confirming = [
            rep for rep in all_reps
            if self._evidence_meets_quorum(evidence, reputation=rep)
        ]
        # Fall back to the bound node's own view only when no peer registry is
        # reachable (isolated tests), matching `_all_reputation_views`' contract.
        if not all_reps and self._evidence_meets_quorum(evidence):
            confirming = [None]

        if confirming:
            self.locally_slashed_nodes.add(target)
            logger.critical(
                f"⛔ [LOCAL_BLOCK] {target} blocked locally by quorum vote "
                f"(weight {evidence.get('approval_weight', 0.0):.1f}, "
                f"{len(evidence.get('approvals', set()))} distinct voters, "
                f"{len(confirming)}/{max(len(all_reps), 1)} peer ledgers confirmed). "
                f"Reputation ledger untouched - will be settled at L1."
            )

    def _penalize_approvers_of_slashed_node(self, target: str, reason: str, reputations: List[Any]) -> None:
        """
        Applies a smaller penalty (apply_approver_penalty) to every distinct
        node that ever approved (referenced as a DAG parent) a block
        authored by `target`, now that `target` has been confirmed malicious
        by consensus. Reuses `approvers_seen`, built for the approval_weight
        collusion fix - a block's approvers ARE its distinct-voter record,
        so no new bookkeeping is needed here.

        `reputations` is the list of peer ledgers that actually confirmed
        quorum on `target` (see `create_commitment_payload`) - each gets the
        penalty applied on its own view, consistent with reputation being
        per-observer rather than one shared ledger. See
        thesis-hub/docs/decisions.md, 2026-09-10.

        Guarded by `self._approver_penalty_applied` so this is safe to call
        more than once for the same target - which of the DAG-derived
        approvers to penalize is global/shared (`approvers_seen`, computed
        once here), only where the penalty lands is per-peer.
        """
        if target in self._approver_penalty_applied:
            return
        self._approver_penalty_applied.add(target)

        penalized: Set[str] = set()
        for block_id, block in self.blocks.items():
            if block.get('proposer') != target:
                continue
            for approver_id in self.approvers_seen.get(block_id, set()):
                if approver_id == target or approver_id in penalized:
                    continue
                for rep in reputations:
                    rep.apply_approver_penalty(approver_id, reason=f"approved_slashed_{target}")
                penalized.add(approver_id)


    def _schedule_gossip(self, block: Dict[str, Any], author: str, simulator: Any) -> None:
        """Schedules gossip delivery using simulator's native broadcast mechanism."""
        if simulator and hasattr(simulator, 'broadcast'):
            latency = simulator._calculate_latency() if hasattr(simulator, '_calculate_latency') else 0.1
            simulator.broadcast(author, block, delay=latency)