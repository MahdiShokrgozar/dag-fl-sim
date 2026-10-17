"""
IronForge Protocol Wrapper Module.

This module acts as the Facade/Factory for the IRONFORGE protocol.
It strictly implements the NetworkProtocol interface and routes the 
simulator to the specific strategies (Aggregation, Selection, Consensus, 
and L2 Settlement) defined for IronForge.
"""

import logging
from typing import Dict, Any

from src.protocols.base import NetworkProtocol
from src.protocols.names import ProtocolNames

# Import IronForge specific strategies
from src.strategies.aggregation.ironforge_aggregator import IronForgeAggregator
from src.strategies.selection.ironforge_selection import IronForgeTipSelector
from src.strategies.consensus.ironforge_vrf import IronForgeConsensus
from src.strategies.l2.ironforge_settlement import IronForgeL2Strategy

# Import default strategies for unused components
from src.strategies.batching.no_batch import NoBatchPolicy
from src.strategies.processing.identity import IdentityProcessor

logger = logging.getLogger("Protocol.IronForge")

class IronForgeProtocol(NetworkProtocol):
    """
    The main protocol class for IRONFORGE.
    Ensures all nodes using this protocol receive the correct strategy components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the protocol and its stateful components (Consensus and L2).
        """
        super().__init__(config)
        self.protocol_name = ProtocolNames.IRONFORGE
        
        # 1. Initialize stateful components as singletons for this protocol instance
        # The VRF Consensus and L2 Settlement must share the exact same state (Ledger)
        self.logger = logging.getLogger("IronForgeProtocol")
        self.logger.info("Initializing IronForge Protocol Components...")

        # Shared across every per-node IronForgeConsensus instance (see
        # get_consensus_strategy() below) so that "was tx_hash challenged"
        # is a single, network-wide fact rather than something each node
        # decides independently for the same block - Algorithm 3's challenge
        # is a one-time event on the block, not a per-observer opinion.
        # See thesis-hub/docs/decisions.md, 2026-09-15.
        self._challenge_registry: Dict[str, bool] = {}

        # Instantiate Consensus Engine (handles VRF and PoL)
        self._consensus = IronForgeConsensus(config, challenge_registry=self._challenge_registry)

        # Instantiate L2 Strategy (handles Settlement and Ledger), injecting the consensus engine
        self._l2_strategy = IronForgeL2Strategy(config, consensus_engine=self._consensus)

    def get_aggregator(self):
        """Returns the weighted aggregator based on E_p."""
        return IronForgeAggregator(self.config)

    def get_tip_selector(self):
        """Returns the Beta/Sigma tip selector with analytical delay."""
        return IronForgeTipSelector(self.config)

    def get_consensus_strategy(self):
        """
        Returns a PER-NODE VRF & PoL consensus engine.

        This used to hand every node the same `self._consensus` singleton. Since
        BaseNode does `consensus.node = self` for each node it constructs, that
        left `self.node` pointing at whichever node happened to be built last,
        so the engine had no idea which node was actually validating. FARABI
        already builds one instance per node for this reason
        (`FarabiProtocol.get_consensus_strategy` returns `ConsensusClass(cfg)`).

        The distinction only started to matter once Algorithm 3's PBFT quorum
        was implemented: the settlement layer counts DISTINCT committee
        verifiers, and with one shared engine every vote carried the same
        identity, so no block ever reached quorum. Measured: "Processing 0
        records from the current epoch" on all 8 settlements, zero rewards and
        zero penalties for the whole run - IronForge's incentive ledger was
        completely inert.

        Shared state stays shared: the settlement ledger, committee and balances
        live in `self._l2_strategy`, and each new engine registers its callback
        with that one instance. See thesis-hub/docs/decisions.md, 2026-09-12.
        `challenge_registry` is shared the same way, for the same reason -
        see 2026-09-15.
        """
        consensus = IronForgeConsensus(self.config, challenge_registry=self._challenge_registry)
        if hasattr(self._l2_strategy, "_on_block_validated"):
            consensus.register_validation_callback(self._l2_strategy._on_block_validated)
        return consensus

    def get_l2_strategy(self):
        """Returns the L2 periodic settlement strategy."""
        return self._l2_strategy

    def get_batch_policy(self):
        """IronForge broadcasts transactions immediately, so no batching is used."""
        return NoBatchPolicy(self.config)

    def get_update_processor(self):
        """No specific payload processing (like compression) is defined in base IronForge."""
        return IdentityProcessor(self.config)

    def check_and_trigger_settlement(self, nodes: dict, current_time: float, simulator: object = None) -> None:
        """
        Called by the DecentralizedSimulator after every MESSAGE_DELIVERY event.
        Delegates to L2Strategy to check if Delta_T has elapsed and run Algorithm 4.
        """
        if self._l2_strategy and hasattr(self._l2_strategy, 'trigger_settlement_if_due'):
            self._l2_strategy.trigger_settlement_if_due(nodes, current_time, simulator)