"""
IronForge Layer-2 Settlement Strategy.

Implements the L2Strategy interface to manage the economic layer of the protocol.
Handles periodic settlement (Delta_T), VRF committee election, ledger updates 
(slashing/rewarding), and anchoring the state to the Layer-1 (Global) DAG.

References:
    - Algorithm 4: Committee Election and Token Settlement.
"""

import logging
from typing import Dict, Any, List

from src.strategies.l2.base import L2Strategy


class IronForgeL2Strategy(L2Strategy):
    """
    IronForge specific implementation of Layer-2 behavior.
    
    Acts as an Observer to the Consensus Engine, listening for PoL validation
    results to accurately reward honest nodes and slash malicious ones.
    """

    def __init__(self, config: Dict[str, Any], consensus_engine: Any):
        """Initializes the L2 Settlement engine and binds to the consensus events."""
        super().__init__(config)
        self.logger = logging.getLogger("IronForge.L2Strategy")
        self.consensus = consensus_engine
        
        # --- Load Configurations ---
        if_cfg = config.get("protocols", {}).get("ironforge", {})
        rep_params = if_cfg.get("reputation", {})
        
        self.settlement_interval = float(rep_params.get("settlement_interval", 50.0))
        self.initial_balance = float(rep_params.get("initial_balance", 10.0))
        self.reward_amount = float(rep_params.get("reward_per_reference", 0.5))
        self.penalty_amount = float(rep_params.get("penalty_amount", 2.0))
        self.slashing_threshold = float(rep_params.get("slashing_threshold", 0.1))
        
        # --- Internal State (Global Ledger) ---
        self.balances: Dict[str, float] = {}
        self.last_settlement_time = 0.0
        self.current_committee: List[Any] = []

        # Tracking validation results during the epoch
        self.epoch_results: List[Dict[str, Any]] = []
        self.slashed_this_epoch: List[str] = []

        # PBFT tallies for Algorithm 3: tx_hash -> {verifier_id: verdict}.
        # Votes for a block are discarded as soon as it reaches quorum, and
        # `_pol_settled` is an insertion-ordered bounded window rather than an
        # unbounded set - a run is open-ended, so anything keyed by block hash
        # has to be capped or it grows for the life of the simulation.
        self._pol_votes: Dict[str, Dict[str, bool]] = {}
        self._pol_settled: Dict[str, bool] = {}
        self._pol_settled_window = int(
            config.get("protocols", {}).get("ironforge", {})
                  .get("consensus", {}).get("pol_settled_window", 512)
        )

        # --- Subscribing to Consensus Events ---
        if hasattr(self.consensus, 'register_validation_callback'):
            self.consensus.register_validation_callback(self._on_block_validated)

    def _on_block_validated(self, author: str, tx_hash: str, is_valid: bool,
                            verifier_id: str = "unknown") -> None:
        """
        Event handler triggered automatically by the Consensus Engine.

        Algorithm 3 settles a PoL verdict by PBFT agreement among the VRF-elected
        verification committee, so this collects votes and records a result only
        once a committee quorum agrees. Previously every verdict from every node
        was appended directly, which meant one verifier's opinion settled a block
        outright -- structurally weaker than FARABI's evidence/vote/quorum path
        and not what the paper specifies, so a head-to-head between them was
        comparing different amounts of consensus rather than different signals.
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        committee_ids = {str(c.node_id) for c in self.current_committee}

        # Before the first committee election there is nobody to form a quorum
        # with; fall back to recording directly so early epochs are not dropped.
        if not committee_ids:
            self.epoch_results.append({
                "tx_hash": tx_hash, "author": author, "is_valid": is_valid
            })
            return

        if verifier_id not in committee_ids:
            return  # Non-committee observers do not get a vote on PoL.

        # Settled check comes FIRST. Doing it after `setdefault` re-created the
        # tally for an already-settled block on every late vote, so the tally
        # dict grew without bound for the rest of the run.
        if tx_hash in self._pol_settled:
            return

        votes = self._pol_votes.setdefault(tx_hash, {})
        if verifier_id in votes:
            return  # One vote per committee member per block.
        votes[verifier_id] = bool(is_valid)

        # PBFT quorum: with n = 3f+1 members the protocol tolerates
        # f = (n-1)//3 faults and commits on 2f+1 matching votes.
        #
        # This was `2*n//3 + 1`, which is the same number for n in {4,7,10} but
        # STRICTER at every other size - at the default committee_size of 5 it
        # demanded 4 of 5 where PBFT asks for 3. Combined with a 5-second
        # settlement interval that clears the tally, blocks were running out of
        # time before they could collect a fourth distinct committee vote:
        # 8 of 8 settlements in seed_43 processed zero records, and the whole
        # run produced one reward and no penalties.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        n_committee = len(committee_ids)
        f_tolerated = (n_committee - 1) // 3
        quorum = min(n_committee, 2 * f_tolerated + 1)
        approvals = sum(1 for v in votes.values() if v)
        rejections = sum(1 for v in votes.values() if not v)

        if approvals >= quorum or rejections >= quorum:
            verdict = approvals >= quorum
            self._pol_settled[tx_hash] = verdict
            self._pol_votes.pop(tx_hash, None)          # tally no longer needed
            if len(self._pol_settled) > self._pol_settled_window:
                self._pol_settled.pop(next(iter(self._pol_settled)))
            self.epoch_results.append({
                "tx_hash": tx_hash, "author": author, "is_valid": verdict
            })
            # info, not debug: "how often does the committee actually
            # reach quorum" is a read-off criterion of plan 24, and at
            # debug level it was invisible in the run logs.
            self.logger.info(
                f"⚖️ PoL quorum on {str(tx_hash)[:8]} by {author}: "
                f"{approvals}approve/{rejections}reject of {len(committee_ids)} "
                f"(quorum {quorum}) -> {'VALID' if verdict else 'INVALID'}"
            )

    def get_balance(self, node_id: str) -> float:
        """Retrieves the current ledger balance for a given node."""
        return self.balances.get(node_id, self.initial_balance)

    # ==========================================
    # 1. Formation (Topology & Roles)
    # ==========================================
    def form_l2_topology(self, nodes: Dict[str, Any], current_time: float) -> None:
        """
        Runs the VRF to elect the committee for the current settlement epoch.
        Updates the L2 state of all physical nodes.
        """
        node_objects = list(nodes.values())
        
        for n in node_objects:
            n.reputation_score = self.get_balance(n.node_id)
            
        self.current_committee = self.consensus.select_committee(node_objects)
        committee_ids = [c.node_id for c in self.current_committee]
        
        self.logger.info(f"🏛️ L2 Topology Formed. Committee: {committee_ids}")
        
        for n in node_objects:
            if not hasattr(n, 'l2_state'):
                n.l2_state = {}
            n.l2_state["role"] = "committee" if n.node_id in committee_ids else "worker"
            n.l2_state["active_session"] = f"epoch_{int(current_time)}"

    # ==========================================
    # 2. Intra-Layer Operation
    # ==========================================
    def select_aggregator(self, cluster_members: List[str], l2_state: Dict) -> str:
        """Returns the committee member with the highest reputation as the leader."""
        if self.current_committee:
            leader = max(self.current_committee, key=lambda c: self.get_balance(c.node_id))
            return leader.node_id
        return cluster_members[0] if cluster_members else "unknown"

    def validate_l2_update(self, update: Dict, l2_state: Dict) -> bool:
        """
        Validates an incoming Task-DAG update using Proof-of-Learning.
        """
        author = update.get("author", update.get("payload", {}).get("author", "unknown"))
        if not author:
            return False
            
        verifier_node = self.current_committee[0] if self.current_committee else None
        
        self.logger.debug(f"⚖️ L2 Committee routing block from {author} to PoL validator...")
        
        # Note: We do NOT manually append to epoch_results here anymore.
        # The validate_block method will trigger _on_block_validated automatically.
        is_valid = self.consensus.validate_block(update, node=verifier_node)
        
        return is_valid

    # ==========================================
    # 3. Settlement (Interaction with L1)
    # ==========================================
    def should_route_to_l2(self, node: Any, transaction: Dict) -> bool:
        """In IronForge, training updates bypass L2 and go directly to L1."""
        return False

    def should_commit_to_l1(self, current_time: float) -> bool:
        """Checks if the Settlement Interval (Delta_T) has elapsed."""
        return (current_time - self.last_settlement_time) >= self.settlement_interval

    def trigger_settlement_if_due(self, nodes: Dict[str, Any], current_time: float, simulator: Any) -> None:
        """
        Executes the full settlement cycle (Algorithm 4) if Delta_T has elapsed.
        """
        if not self.should_commit_to_l1(current_time):
            return

        self.logger.info(f"⏰ Delta_T elapsed at t={current_time:.2f}s — running Settlement (Algorithm 4)")

        self.form_l2_topology(nodes, current_time)

        settlement_payload = self.create_commitment_payload(
            aggregated_model=None,
            l2_metadata={"current_time": current_time, "nodes_ref": nodes}
        )

        if hasattr(simulator, 'dag') and settlement_payload:
            settlement_tx = {
                "id": f"settlement_{current_time:.2f}",
                "hash": f"sh_{current_time:.2f}",
                "timestamp": current_time,
                "type": "SettlementNode",
                "parents": list(simulator.dag.tips)[:2] if simulator.dag.tips else [],
                "payload": settlement_payload,
                "author": "committee",
            }
            simulator.dag.add_transaction(settlement_tx)
            self.logger.info(f"⚓ Settlement node anchored to Global-DAG: {settlement_tx['hash']}")

    def create_commitment_payload(self, aggregated_model: Any, l2_metadata: Dict) -> Dict[str, Any]:
        """
        Resolves the epoch, applies token penalties/rewards, updates physical nodes,
        and generates the final Settlement Node payload.
        """
        current_time = l2_metadata.get("current_time", self.last_settlement_time + self.settlement_interval)
        nodes_ref = l2_metadata.get("nodes_ref", {})
        self.last_settlement_time = current_time
        
        settlement_report = []

        self.logger.info(f"💰 --- SETTLEMENT PHASE INITIATED (Epoch {current_time:.2f}s) ---")
        self.logger.info(f"📊 Processing {len(self.epoch_results)} records from the current epoch.")

        # --- Apply Ledger Updates ---
        for res in self.epoch_results:
            author = res["author"]
            if author not in self.balances:
                self.balances[author] = self.initial_balance

            old_balance = self.balances[author]

            if res["is_valid"]:
                self.balances[author] += self.reward_amount
                settlement_report.append((res["tx_hash"], "approved"))
                self.logger.info(f"📈 Rewarded {author}: {old_balance:.2f} -> {self.balances[author]:.2f}")
            else:
                self.balances[author] -= self.penalty_amount
                settlement_report.append((res["tx_hash"], "rejected"))
                self.logger.warning(f"📉 Penalized {author}: {old_balance:.2f} -> {self.balances[author]:.2f}")
                
                if self.balances[author] <= self.slashing_threshold:
                    if author not in self.slashed_this_epoch:
                        self.slashed_this_epoch.append(author)
                        self.logger.error(f"☠️ SLASHED {author}! Balance dropped to {self.balances[author]:.2f}")

        # --- Sync Ledger to Physical Nodes (for CSV Reporting) ---
        for node_id, node_obj in nodes_ref.items():
            if node_id in self.balances:
                node_obj.reputation_score = self.balances[node_id]
                if node_id in self.slashed_this_epoch or self.balances[node_id] <= self.slashing_threshold:
                    node_obj.is_slashed = True
                    if hasattr(node_obj, 'l2_state'):
                        node_obj.l2_state["role"] = "slashed"
                        
        self.logger.info("💰 --- SETTLEMENT PHASE COMPLETED ---")
        
        # --- Prepare Payload ---
        commitment_payload = {
            "type": "SettlementNode",
            "timestamp": current_time,
            "committee": [c.node_id for c in self.current_committee] if self.current_committee else [],
            "report": settlement_report,
            "slashed_nodes": list(self.slashed_this_epoch),
            "ledger_state_hash": hash(frozenset(self.balances.items()))
        }
        
        # Cleanup for next epoch. Unfinished PoL tallies are dropped with the
        # epoch they belong to: the committee is re-elected each settlement
        # (Algorithm 4), so votes cast by the outgoing committee cannot be
        # completed by the incoming one and would otherwise accumulate forever.
        # `_pol_settled` survives -- it is what stops a late vote from settling
        # the same block twice -- and is bounded by its own window.
        self.epoch_results.clear()
        self.slashed_this_epoch.clear()
        self._pol_votes.clear()
        
        return commitment_payload