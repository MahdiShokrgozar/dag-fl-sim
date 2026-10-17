# src/protocols/generic.py

from typing import Any, Optional
from src.protocols.base import NetworkProtocol
class GenericProtocol(NetworkProtocol):
    """
    A generic protocol implementation that composes behavior using plugged-in strategies.
    
    Instead of hardcoding logic (like in IronForge or VanillaFL), this protocol 
    delegates decisions to the strategies injected via the Factory.
    """

    def __init__(
        self,
        config: Any,
        aggregator: Any = None,
        consensus: Any = None,
        tip_selector: Any = None,
        batch_policy: Any = None,
        update_processor: Any = None,
        l2_strategy: Any = None
    ):
        super().__init__(config)
        
        # Dependency Injection: Store the strategies
        self.aggregator = aggregator
        self.consensus = consensus
        self.tip_selector = tip_selector
        self.batch_policy = batch_policy
        self.update_processor = update_processor
        self.l2_strategy = l2_strategy
        
    # --- Strategy Accessors (Used by Nodes) ---

    def get_aggregator(self):
        return self.aggregator
    
    def get_tip_selector(self):
        return self.tip_selector

    def get_consensus_strategy(self):
        return self.consensus

    def get_batch_policy(self):
        return self.batch_policy
        
    def get_update_processor(self):
        return self.update_processor

    def get_l2_strategy(self):
        return self.l2_strategy