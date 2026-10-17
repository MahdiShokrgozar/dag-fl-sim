# src/protocols/dbfl.py
from typing import Dict, Any

from .base import NetworkProtocol
from src.strategies.aggregation.weighted_fedavg import WeightedFedAvg
from src.strategies.batching.size_based import CountBasedBatching
from src.strategies.consensus.simple import SimpleConsensus # DB-FL uses DAG validation implicitly
from src.strategies.l2.clustering import RandomClustering

# DB-FL Specifics
from src.strategies.reputation.dbfl import DBFLReputation
from src.strategies.selection.rtsa import RTSATipSelector
from src.strategies.processing.dropout import ReputationBasedDropout

class DBFLProtocol(NetworkProtocol):
    """
    Protocol definition for 'DB-FL: DAG blockchain-enabled generalized federated dropout learning'.
    Integrates:
    - Entropy-based Reputation
    - RTSA Tip Selection
    - Federated Dropout
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Shared Reputation Ledger (The "Blockchain" state)
        self.reputation_ledger = DBFLReputation(config)

    def get_aggregator(self):
        # DB-FL uses standard weighted aggregation (Eq 4)
        return WeightedFedAvg(self.config)

    def get_tip_selector(self):
        # RTSA Algorithm
        return RTSATipSelector(self.reputation_ledger, self.config)

    def get_consensus_strategy(self):
        # DB-FL relies on DAG structure for security, standard validation
        return SimpleConsensus(self.config)

    def get_batch_policy(self):
        return CountBasedBatching(self.config)

    def get_update_processor(self):
        # Apply Reputation-based Dropout
        processor = ReputationBasedDropout(self.config, self.reputation_ledger)
        return processor

    def get_l2_strategy(self):
        # Can be enabled if we want hierarchical DB-FL (ES vs IDs)
        # Using RandomClustering to simulate ES-ID association
        return None#RandomClustering(self.config)

    def get_reputation_module(self):
        # Without this override, BaseNode.get_current_reputation() (which the
        # metrics pipeline reads reputation_score through) falls through to
        # its generic 1.0 default for every DB-FL node, regardless of what
        # the entropy-weighted reputation module actually computes -
        # DBFLReputation.is_quarantined()/get_reputation_score() exist
        # specifically to make this override safe. See
        # thesis-hub/docs/decisions.md, 2026-09-15.
        return self.reputation_ledger