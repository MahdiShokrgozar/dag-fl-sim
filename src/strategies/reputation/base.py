# src/strategies/reputation/base.py

from abc import ABC, abstractmethod
import logging

class BaseReputation(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_reputation_score(self, node_id: str) -> float:
        """
        Unified interface to retrieve the normalized reputation score of a node.
        Every protocol MUST implement this method.
        """
        pass
    
    def is_quarantined(self, node_id: str) -> bool:
        return False
