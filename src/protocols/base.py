# src/protocols/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

# فقط برای تایپ‌هینت‌ها ایمپورت می‌کنیم (در زمان اجرا ایمپورت نمی‌شوند)
if TYPE_CHECKING:
    from src.strategies.aggregation.base import BaseAggregator
    from src.strategies.l2.base import L2Strategy
    from src.strategies.selection.base import TipSelector
    from src.strategies.consensus.base import ConsensusStrategy
    from src.strategies.batching.base import BatchingPolicy
    from src.strategies.processing.base import UpdateProcessor

class NetworkProtocol(ABC):
    """
    The Pure Abstract Protocol Contract.
    defines WHICH strategies a node should use, but implementation details
    are left to specific protocol subclasses (e.g., IronForgeProtocol).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # --- Mandatory Components (Must be implemented by subclasses) ---

    @abstractmethod
    def get_aggregator(self) -> 'BaseAggregator':
        pass

    @abstractmethod
    def get_tip_selector(self) -> 'TipSelector':
        pass
    
    # --- Optional Components (Can return None or need Implementation) ---

    @abstractmethod
    def get_consensus_strategy(self) -> 'ConsensusStrategy':
        """Must return a Consensus Strategy (even if it's a dummy/simple one)."""
        pass

    @abstractmethod
    def get_batch_policy(self) -> 'BatchingPolicy':
        pass

    @abstractmethod
    def get_update_processor(self) -> 'UpdateProcessor':
        pass

    def get_l2_strategy(self) -> Optional['L2Strategy']:
        """Returns None by default (Flat Network). Override if L2 is needed."""
        return None
    
    def get_reputation_module(self) -> Optional[Any]:
        """Returns None by default. Override if the protocol includes a reputation system."""
        return None