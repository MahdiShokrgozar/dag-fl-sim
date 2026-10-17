# src/strategies/reputation/dbfl.py
import numpy as np
import math
import logging
from typing import Dict, Any, List
from src.strategies.base import Reputation

logger = logging.getLogger(__name__)

class DBFLReputation(Reputation):
    """
    Entropy Weight-based Reputation Mechanism (DB-FL Paper, Section 2.2).
    
    This class evaluates Edge Server (ES) quality by calculating an entropy-based
    score using 5 distinct indicators:
        1. Historical Reputation
        2. Total Local Training Rounds
        3. Local Dataset Size
        4. Cosine Similarity of the Edge Model
        5. Edge Model Test Accuracy
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.rep_config = config.get("protocols", {}).get("dbfl", {}).get("reputation", {})
        
        self.initial_score = float(self.rep_config.get("initial_score", 50.0))
        self.epsilon = float(self.rep_config.get("epsilon", 1e-9))
        self.alpha = float(self.rep_config.get("alpha", 0.5))
        
        # Mapping of node_id to its final calculated reputation score
        self.scores: Dict[str, float] = {}
        # Mapping of node_id to its raw metric history
        self.history: Dict[str, Dict[str, float]] = {}

    def update(self, node_id: str, metric_data: Dict[str, Any]) -> None:
        """
        Updates the internal metrics history for a given node and triggers
        the entropy recalculation process.
        
        Args:
            node_id: Unique identifier for the node (ES or ID).
            metric_data: Dictionary containing the latest raw metrics.
        """
        # Fallback to logical defaults if specific metric data is missing
        current_metrics = {
            "hist_rep": self.scores.get(node_id, self.initial_score), 
            "rounds": float(metric_data.get("rounds", 1.0)),     
            "data_size": float(metric_data.get("samples", 1.0)), 
            "similarity": float(metric_data.get("similarity", 1.0)), 
            "accuracy": float(metric_data.get("accuracy", 0.0))      
        }
        
        # Apply Exponential Moving Average (EMA) to prevent drastic fluctuations
        if node_id in self.history:
            for key in current_metrics:
                self.history[node_id][key] = (self.alpha * current_metrics[key]) + \
                                             ((1 - self.alpha) * self.history[node_id][key])
        else:
            self.history[node_id] = current_metrics
            
        self._recalculate_entropy_scores()

    def get_score(self, node_id: str) -> float:
        """
        Retrieves the current reputation score of a node.
        Defaults to the initial_score if the node is unseen.
        """
        return self.scores.get(node_id, self.initial_score)

    def get_reputation_score(self, node_id: str) -> float:
        """
        Alias for get_score(). BaseNode.get_current_reputation() (the facade
        the metrics pipeline reads reputation_score through) specifically
        looks for this method name, not get_score - without it, every DB-FL
        node's reported reputation_score column silently read the generic
        fallback of 1.0 regardless of this module's real, live score. See
        thesis-hub/docs/decisions.md, 2026-09-15.
        """
        return self.get_score(node_id)

    def is_quarantined(self, node_id: str) -> bool:
        """
        DB-FL (Sec. 2.2-2.5) uses reputation to bias tip selection and set a
        dropout rate - it never describes excluding a node outright. Always
        False, matching the same "no quarantine concept -> skip the check
        rather than invent one" precedent already established for IronForge
        (see thesis-hub/docs/decisions.md, 2026-09-12). Required because
        BaseNode.create_and_broadcast_block calls this unconditionally once
        `self.reputation` is not None.
        """
        return False

    def _recalculate_entropy_scores(self) -> None:
        """
        Core algorithm implementing Equations (5) through (9) from the paper.
        Calculates normalized entropy weights for all tracked indicators and
        updates the final reputation scores globally.
        """
        nodes = list(self.history.keys())
        n_nodes = len(nodes)
        
        # Entropy requires at least 2 nodes to be meaningful. Retain current/initial scores.
        if n_nodes < 2:
            for node in nodes:
                self.scores[node] = self.initial_score
            return

        indicators = ["hist_rep", "rounds", "data_size", "similarity", "accuracy"]
        matrix = np.zeros((n_nodes, len(indicators)))

        # Construct the raw observation matrix
        for i, node in enumerate(nodes):
            for j, key in enumerate(indicators):
                matrix[i, j] = self.history[node][key]

        # 1. Standardization (Eq. 6)
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        ranges = maxs - mins
        
        # Prevent division by zero for columns with constant values across all nodes
        ranges[ranges == 0] = 1.0 
        
        normalized_matrix = (matrix - mins) / ranges

        # 2. Entropy Calculation (Eq. 5 & 8)
        # Adding epsilon to denominator to avoid division by zero
        P = normalized_matrix / (normalized_matrix.sum(axis=0) + self.epsilon)
        
        k = 1.0 / math.log(n_nodes)
        
        # Calculate P * ln(P) while safely handling P=0
        temp_P = P.copy()
        temp_P[temp_P == 0] = 1.0 # ln(1) = 0, effectively neutralizing zero-probability elements
        E = -k * (P * np.log(temp_P)).sum(axis=0)
        
        # 3. Entropy Weights (Eq. 8)
        d = 1.0 - E
        weights = d / (d.sum() + self.epsilon)

        # 4. Final Score Calculation (Eq. 9)
        final_scores = (normalized_matrix * weights).sum(axis=1) * 100

        # Clip final scores to strictly remain within the valid [0.0, 100.0] range
        for i, node in enumerate(nodes):
            self.scores[node] = float(np.clip(final_scores[i], 0.0, 100.0))