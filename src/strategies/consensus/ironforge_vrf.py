"""
IronForge VRF Consensus and Proof-of-Learning (PoL) Module.

This module unifies the Consensus Strategy for IronForge. It contains:
1. VRFEngine: A stateless HMAC-based simulator for verifiable randomness.
2. IronForgeConsensus: The main strategy that proposes blocks, validates 
   them via a Ground-Truth Oracle (simulating the Proof-of-Learning mechanism), 
   and elects the settlement committee using cryptographic sortition.

References:
    - Algorithm 3: Proof-of-Learning (PoL) verification (Oracle Simulated).
    - Equation (4): Cryptographic sortition for committee election.
"""

import hashlib
import hmac
import logging
import random
from typing import Dict, Any, List, Tuple, Callable

from src.strategies.consensus.base import ConsensusStrategy


class VRFEngine:
    """
    Lightweight VRF (Verifiable Random Function) simulation using HMAC-SHA256.
    Generates verifiable randomness tied to a node's secret key.
    """
    
    @staticmethod
    def vrf_prove(secret_key: bytes, seed: bytes) -> Tuple[str, str]:
        """Generates the VRF output and its corresponding proof."""
        output = hmac.new(secret_key, seed, hashlib.sha256).hexdigest()
        proof = hmac.new(secret_key, output.encode(), hashlib.sha256).hexdigest()
        return output, proof

    @staticmethod
    def vrf_verify(public_key: bytes, seed: bytes, output: str, proof: str) -> bool:
        """Verifies the authenticity of a VRF output against a public key."""
        expected_output = hmac.new(public_key, seed, hashlib.sha256).hexdigest()
        if expected_output != output:
            return False
        expected_proof = hmac.new(public_key, output.encode(), hashlib.sha256).hexdigest()
        return expected_proof == proof


class IronForgeConsensus(ConsensusStrategy):
    """
    Consensus strategy for the IronForge protocol.
    
    Implements the Observer Pattern to broadcast validation results to any 
    interested listeners (e.g., Layer-2 Settlement Strategies) without 
    tightly coupling the components.
    """

    def __init__(self, config: Dict[str, Any], ledger: Any = None, challenge_registry: Any = None):
        """Initializes the IronForge consensus engine."""
        super().__init__(config)
        self.logger = logging.getLogger("IronForge.Consensus")
        self.vrf_engine = VRFEngine()

        # Maximum possible value for a SHA-256 hash (used for Hash Ring mapping)
        self.MAX_HASH = float(int("F" * 64, 16))

        # Load configurations dynamically
        ironforge_cfg = self.config.get("protocols", {}).get("ironforge", {})
        consensus_cfg = ironforge_cfg.get("consensus", {})
        reputation_cfg = ironforge_cfg.get("reputation", {})

        self.committee_size = int(reputation_cfg.get("committee_size", 5))

        # Oracle hyperparameters simulating the PoL Frobenius norm thresholds
        self.oracle_eps_pol = float(consensus_cfg.get("eps_pol", 2.0))
        self.oracle_min_learning = float(consensus_cfg.get("min_learning_dist", 0.1))

        # Algorithm 3 is a challenge-response game: a published node stands
        # unverified unless some worker spends a deposit to raise a PoL
        # challenge against it - it is not a blanket check run by every
        # observer on every block. The paper gives no numeric challenge rate
        # (it falls out of each worker's economic incentive, which this
        # simulator doesn't model in that detail); this is our own knob.
        # Shared across every per-node instance via `challenge_registry` (see
        # src/protocols/ironforge.py) so the same tx_hash resolves to the
        # same "was it challenged" fact everywhere, keeping the PBFT quorum
        # tally in _on_block_validated meaningful - if each node rolled its
        # own dice independently, committee members could disagree on
        # whether a vote was even cast for the same block.
        # See thesis-hub/docs/decisions.md, 2026-09-15.
        self.challenge_probability = float(consensus_cfg.get("challenge_probability", 0.5))
        self._challenge_registry: Dict[str, bool] = challenge_registry if challenge_registry is not None else {}

        # List of subscribed observers (e.g., L2 Strategy) for validation events
        self._validation_callbacks: List[Callable[[str, str, bool], None]] = []

    def _is_challenged(self, tx_hash: str) -> bool:
        """
        Resolves (once per tx_hash, cached) whether this block was ever
        challenged. Unchallenged blocks are never checked - see
        `validate_block`'s first line.
        """
        decision = self._challenge_registry.get(tx_hash)
        if decision is None:
            decision = random.random() < self.challenge_probability
            self._challenge_registry[tx_hash] = decision
        return decision

    def register_validation_callback(self, callback: Callable[[str, str, bool], None]) -> None:
        """Registers a listener for Proof-of-Learning validation events."""
        if callback not in self._validation_callbacks:
            self._validation_callbacks.append(callback)

    def _broadcast_validation_result(self, author: str, tx_hash: str, is_valid: bool) -> None:
        """
        Notifies all registered observers about the outcome of a block validation.

        Carries the identity of the verifier that reached this verdict. Algorithm 3
        settles PoL by PBFT agreement among the VRF-elected verification committee,
        so the settlement layer has to know WHO voted before it can count a quorum;
        without the verifier id every verdict looked equally authoritative and a
        single node's opinion settled the block.
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        verifier_id = str(getattr(getattr(self, "node", None), "node_id", "unknown"))
        for callback in self._validation_callbacks:
            try:
                try:
                    callback(author, tx_hash, is_valid, verifier_id)
                except TypeError:
                    # Tolerate listeners still on the 3-argument signature.
                    callback(author, tx_hash, is_valid)
            except Exception as e:
                self.logger.error(f"Error executing validation callback for {author}: {e}")

    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """Wraps the transaction payload with IronForge consensus metadata."""
        return {
            "payload": payload,
            "author": node_id,
            "metadata": {
                "consensus_type": "ironforge_pol",
                "pol_claim": "valid_learning" 
            }
        }

    def validate_block(self, block: Dict[str, Any], node: Any = None) -> bool:
        """
        Executes an Oracle-Simulated Proof-of-Learning (PoL) verification (Algorithm 3).

        Instead of invoking heavy PyTorch tensor operations (which cripples the DES),
        this method reads the Ground Truth Oracle to verify if the claimed model 
        trajectory aligns with the true learning capability of the node.
        
        Args:
            block (Dict): The transaction block containing weights and Oracle metadata.
            node (Any): The verifier node.

        Returns:
            bool: Is_Valid. (Validation cost, when a challenge actually happens,
            is charged internally via node.telemetry - not returned; nothing
            consumed the second tuple element anyway, see
            thesis-hub/docs/decisions.md, 2026-09-15.)
        """
        payload = block.get("payload", {})
        author = block.get("author", payload.get("author", "unknown"))
        tx_hash = block.get("hash", "unknown")

        # --- 0. Challenge gate (Algorithm 3) ---
        # Most published nodes are never challenged and simply stand as-is -
        # PoL is a dispute mechanism, not a mandatory per-block inspection.
        # No cost is charged and no vote is cast for an unchallenged block:
        # it is genuinely unverified, not "verified and passed". See the
        # class docstring for `_is_challenged` and thesis-hub/docs/decisions.md,
        # 2026-09-15.
        if not self._is_challenged(tx_hash):
            return True

        # The main path (BaseNode.receive_transaction) calls this with no `node`,
        # while the L2 committee path passes one explicitly. Falling back to the
        # per-node binding that BaseNode already installs (`consensus.node = self`,
        # the same mechanism FarabiConsensus relies on) matters for correctness,
        # not just tidiness: with node=None the telemetry guard below silently
        # skipped, so IronForge's PoL replay cost -- the entire basis of the
        # cost comparison against FARABI -- was never charged on the path that
        # actually runs, and node_capacity was pinned at 1.0 regardless of the
        # node's real capability. See thesis-hub/docs/decisions.md, 2026-09-12.
        node = node if node is not None else getattr(self, "node", None)

        # --- 1. Compute Virtual Execution Latency ---
        # Deliberately analytical rather than an actual retraining pass: the
        # simulator models PoL's cost instead of paying it, so that a fixed
        # hardware budget can still compare protocols on equal accounting.
        # `pol_replay_time_factor` is IronForge's own reported replay overhead.
        base_time = self.config.get("simulation", {}).get("base_training_time_per_epoch", 1.0)
        pol_factor = self.config.get("protocols", {}).get("ironforge", {}).get("consensus", {}).get("pol_replay_time_factor", 0.3)
        node_capacity = getattr(node, "capacity", 1.0) if node else 1.0
        delay_multiplier = getattr(node, "delay_multiplier", 1.0) if node else 1.0
        effective_capacity = node_capacity / max(0.01, delay_multiplier)

        validation_delay = (base_time * pol_factor) / max(0.1, effective_capacity)

        # Register the physical delay in the DES graph
        if node is not None and hasattr(node, 'telemetry'):
            node.telemetry.register_compute(validation_delay)

        # --- 2. Sanity Checks ---
        if not payload:
            self.logger.warning(f"🚨 PoL REJECTED [{author}]: Empty payload.")
            self._broadcast_validation_result(author, tx_hash, False)
            return False

        # --- 3. Extract Oracle & Claimed Metrics ---
        # Claimed accuracy is what the malicious node broadcasts to fool the network
        claimed_acc = payload.get("accuracy", 0.0)

        # True accuracy is extracted from the simulation Oracle (SSOT). The oracle
        # stands in for what a real PoL replay would establish, so the simulator
        # does not have to pay for the replay itself - the cost is charged above.
        #
        # Two lookup sources, in the same order FarabiConsensus uses: the
        # simulator-level `oracle_state` keyed by block hash is authoritative,
        # with the in-block copy as fallback. Reading only the in-block copy was
        # fragile - BaseNode builds a stripped `broadcast_tx` intending to keep
        # ground truth off the wire, and if that intent is ever actually wired
        # up this lookup would start returning {} silently.
        #
        # On a miss we now FAIL CLOSED. The previous default of `claimed_acc`
        # made `deviation = |claimed - claimed| = 0`, i.e. every block passed
        # PoL unconditionally - a baseline that detects nothing while appearing
        # to work. An unverifiable block is not a verified one.
        oracle = {}
        sim = getattr(node, "simulator", None) if node is not None else None
        if sim is not None:
            oracle = getattr(sim, "oracle_state", {}).get(tx_hash, {}) or {}
        ground_truth = oracle or block.get("ground_truth", {})

        if not ground_truth:
            self.logger.warning(
                f"🚨 PoL UNVERIFIABLE [{author}]: no ground-truth record for {str(tx_hash)[:8]}. "
                f"Rejecting rather than defaulting to the claim."
            )
            self._broadcast_validation_result(author, tx_hash, False)
            return False

        true_acc = ground_truth.get("accuracy", 0.0)
        parent_acc = ground_truth.get("parent_accuracy", 0.0)

        # --- 4. Simulated Anti-Lazy Firewall (Equation equivalent to IronForge min_dist) ---
        # Lazy nodes duplicate base weights, meaning their true accuracy won't improve
        learning_dist = true_acc - parent_acc
        if learning_dist < self.oracle_min_learning and true_acc < 90.0:
            # We allow small learning dist if the model is already highly converged (>90%)
            self.logger.warning(
                f"🚨 PoL REJECTED [{author}]: LAZY attack detected. "
                f"Learning dist ({learning_dist:.2f}) < threshold ({self.oracle_min_learning})."
            )
            self._broadcast_validation_result(author, tx_hash, False)
            return False

        # --- 5. Simulated PoL Replay Evaluation (Equation equivalent to epsilon_PoL) ---
        # An honest replay would yield the true accuracy. If the claimed accuracy deviates
        # from the true capability by more than eps_pol, the node forged the weights.
        deviation = abs(claimed_acc - true_acc)
        
        is_valid = deviation <= self.oracle_eps_pol

        if is_valid:
            self.logger.debug(f"✅ PoL PASSED [{author}]: Deviation {deviation:.2f} <= {self.oracle_eps_pol}")
        else:
            self.logger.warning(
                f"🚨 PoL REJECTED [{author}]: Forged weights detected! "
                f"Deviation ({deviation:.2f}) > eps_pol ({self.oracle_eps_pol}). "
                f"(Claimed: {claimed_acc:.1f}, True: {true_acc:.1f})"
            )

        # --- 6. Broadcast Result ---
        self._broadcast_validation_result(author, tx_hash, is_valid)
        
        return is_valid

    def select_committee(self, active_nodes: List[Any], epoch_id: str = "epoch_0", last_hash: str = "genesis") -> List[Any]:
        """
        Executes cryptographic sortition based on Equation (4) of IronForge.
        Condition: VRF-hash(prv_k, seed) < Lambda_k (Proportional to Reputation).
        """
        if not active_nodes:
            return []

        seed_string = f"{epoch_id}|{last_hash}"
        seed_bytes = hashlib.sha256(seed_string.encode()).digest()
        
        total_balance = sum(getattr(n, 'reputation_score', 1.0) for n in active_nodes)
        total_balance = max(total_balance, 1e-9) 
            
        eligible_nodes = []
        
        for node in active_nodes:
            # Using node_id as deterministic mock secret key
            sk = str(node.node_id).encode('utf-8')
            balance = getattr(node, 'reputation_score', 1.0)
            
            lambda_k = (balance / total_balance) * self.MAX_HASH
            output, _ = self.vrf_engine.vrf_prove(sk, seed_bytes)
            vrf_int = int(output, 16)
            
            if vrf_int < lambda_k:
                eligible_nodes.append({"node": node, "vrf_int": vrf_int})
                
        # Sort eligible nodes by VRF integer to rank the "luckiest" ones
        eligible_nodes.sort(key=lambda x: x["vrf_int"])
        winners = [item["node"] for item in eligible_nodes[:self.committee_size]]
        
        # Fallback mechanism if cryptography yields no winners due to strict thresholds
        if not winners:
            active_nodes_sorted = sorted(active_nodes, key=lambda n: getattr(n, 'reputation_score', 1.0), reverse=True)
            winners = active_nodes_sorted[:self.committee_size]
            
        return winners