# src/strategies/l2/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

class L2Strategy(ABC):
    """
    Abstract Base Class defining a complete Layer-2 Protocol logic.
    Layer 2 is not just topology; it encompasses formation, execution, and settlement.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # --- 1. Formation (Topology & Roles) ---
    @abstractmethod
    def should_route_to_l2(self, node: Any, payload: Dict[str, Any]) -> bool:
        """
        Determines if the payload should be routed to the L2 network.
        Must be implemented by specific L2 strategies.
        """
        pass
    
    @abstractmethod
    def form_l2_topology(self, nodes: Dict[str, Any], current_time: float) -> None:
        """
        Groups nodes, assigns roles (Head/Worker/Verifier), and establishes channels.
        Example: Cluster nodes based on geolocation or reputation.
        """
        pass

    # --- 2. Intra-Layer Operation (Internal Consensus) ---
    @abstractmethod
    def select_aggregator(self, cluster_members: List[str], l2_state: Dict) -> str:
        """
        Determines WHO aggregates updates inside the L2 instance.
        Example: Round-robin, Reputation-based, or PoW winner.
        """
        pass

    @abstractmethod
    def validate_l2_update(self, update: Dict, l2_state: Dict) -> bool:
        """
        Specific consensus rules INSIDE Layer 2 (might differ from L1).
        Example: Check signature against cluster head only.
        """
        return True

    # --- 3. Settlement (Interaction with L1) ---
    @abstractmethod
    def should_commit_to_l1(self, current_time: float) -> bool:
        """
        Decides WHEN to settle the L2  back to the main chain (L1).
        Example: After 10 rounds, or when accuracy > 90%.
        """
        pass

    @abstractmethod
    def create_commitment_payload(self, aggregated_model: Any, l2_metadata: Dict) -> Dict[str, Any]:
        """
        Formats the data for the L1 Ledger.
        Example: Creating a 'Rollup' proof or a simple checkpoint hash.
        """
        pass