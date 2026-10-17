# src/strategies/selection/rtsa.py
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Union
from src.strategies.selection.base import TipSelector
from src.strategies.reputation.dbfl import DBFLReputation

logger = logging.getLogger(__name__)

class RTSATipSelector(TipSelector):
    """
    Reputation-based Tips Selection Algorithm (RTSA).
    Selects tips based on Cumulative Weight (CW) and ES Reputation.
    Reference: DB-FL Paper, Section 2.3, Eq (11).
    Probability P_xy ~ exp(kappa * (CW_y + R_y))
    """
    def __init__(self, reputation_module: DBFLReputation, config: Dict[str, Any]):
        super().__init__(config)
        self.reputation = reputation_module
        
        # Extract configuration dynamically without hardcoding values
        dbfl_config = config.get("protocols", {}).get("dbfl", {}).get("consensus", {})
        self.kappa = float(dbfl_config.get("kappa", 0.5))
        self.num_tips = int(dbfl_config.get("num_tips", 2))

    def _calculate_cumulative_weight(self, dag: Any, tx_hash: str) -> float:
        """
        Calculates the Cumulative Weight (CW) of a transaction.
        In a standard DAG, CW = base_weight (1) + sum of weights of all future approvers.
        Since true 'tips' have no approvers yet, their CW is inherently 1.0.
        This method dynamically checks for approvers to support delayed or lazy tips.
        """
        # Base weight for any valid transaction is 1.0
        cw = 1.0
        
        # O(1) Optimization: If the transaction is explicitly in dag.tips, 
        # it is guaranteed to have no children (approvers) yet.
        if tx_hash in dag.tips:
            return cw
            
        # O(N) Fallback: Count direct approvers if the transaction is not a pure tip.
        # Note: If dag.py implements a reverse-lookup (children dict), this could be O(1).
        for vertex_data in dag.vertices.values():
            if tx_hash in vertex_data.get('parents', []):
                cw += 1.0 
                
        return cw

    def select_tips(self, dag: Any, node: Any = None) -> Union[List[str], Tuple[List[str], float]]:
        """
        Selects exactly 'num_tips' from the DAG using the RTSA probability distribution.
        """
        tips = list(dag.tips)
        
        # Fallback to Genesis if no tips exist in the network
        if not tips:
            genesis_hash = dag.genesis['hash'] if hasattr(dag, 'genesis') else '0'*64
            return [genesis_hash]
        
        # If available tips are less than or equal to the required amount, return all of them
        if len(tips) <= self.num_tips:
            return tips

        tip_scores = []
        valid_tips = []
        
        for tip_hash in tips:
            tx = dag.vertices.get(tip_hash)
            if not tx:
                continue
            
            # 1. Retrieve Reputation (R_y)
            # Ensure we fallback to 'unknown' if creator metadata is missing
            creator = tx.get("creator", "unknown")
            if isinstance(tx.get("update"), dict) and "_original_creator_id" in tx["update"]:
                creator = tx["update"]["_original_creator_id"]
                
            rep_score = self.reputation.get_score(creator)
            
            # 2. Calculate Cumulative Weight (CW_y)
            cw = self._calculate_cumulative_weight(dag, tip_hash)
            
            # 3. Combine Scores (CW_y + R_y)
            # Higher CW & Reputation -> Higher selection probability
            score = cw + rep_score
            
            tip_scores.append(score)
            valid_tips.append(tip_hash)

        if not valid_tips:
            return [dag.genesis['hash']]

        # Softmax calculation for probabilities (Adjusted Eq 11 for standard logic)
        scores_array = np.array(tip_scores, dtype=np.float64)
        
        # Numerical stability: subtract max to prevent np.exp() overflow
        scores_array = scores_array - np.max(scores_array)
        
        # Apply kappa coefficient
        probs = np.exp(self.kappa * scores_array)
        
        # Normalize probabilities to sum exactly to 1.0
        prob_sum = probs.sum()
        if prob_sum == 0 or np.isnan(prob_sum):
            # Fallback to uniform distribution if mathematical error occurs (e.g., extreme values)
            logger.warning("RTSA softmax failed (sum=0 or NaN). Falling back to uniform distribution.")
            probs = np.ones(len(valid_tips)) / len(valid_tips)
        else:
            probs = probs / prob_sum
        
        # Select tips based on calculated probabilities without replacement
        selected_indices = np.random.choice(
            len(valid_tips), 
            size=self.num_tips, 
            p=probs, 
            replace=False
        )
        
        selected_tips = [valid_tips[i] for i in selected_indices]
        
        return selected_tips