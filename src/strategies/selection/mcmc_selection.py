# src/strategies/selection/mcmc_selection.py

import random
import logging
from typing import List, Any, Dict, Set
from .base import TipSelector

logger = logging.getLogger("TipSelector.MCMC")

class MCMCSelection(TipSelector):
    """
    Markov Chain Monte Carlo (MCMC) Tip Selection Strategy.
    Simulates random walks on the DAG to find tips.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.alpha = self.config.get("network", {}).get("tip_selection_alpha", 1.5)
        self.num_tips = int(config.get("protocol", {}).get("num_tips", 5))

        # Cache for weights to avoid repeated calculations
        self.weights_cache: Dict[str, int] = {}
        
        logger.info(f"MCMC strategy initialized with alpha = {self.alpha}")

    def _calculate_cumulative_weight(self, tx_hash: str, approvers: Dict[str, List[str]], visited: Set[str]):
        """
        Recursively calculates cumulative weight (Future Set Size).
        """
        if tx_hash in self.weights_cache:
            return self.weights_cache[tx_hash]
        
        weight = 1 # Initial weight
        
        if tx_hash in approvers:
            for approver_hash in approvers[tx_hash]:
                # Prevent loops in the DAG
                if approver_hash not in visited:
                    visited.add(approver_hash)
                    weight += self._calculate_cumulative_weight(approver_hash, approvers, visited)
        
        self.weights_cache[tx_hash] = weight
        return weight

    def select_tips(self, dag: Any, node: Any = None) -> List[str]:
        """
        Selects 'self.num_tips' tips using MCMC random walk.

        BaseNode.create_and_broadcast_block calls select_tips(dag, node=self)
        - this used to take a `count` kwarg instead, so every call from the
        live simulator raised TypeError: unexpected keyword argument 'node'.
        See random_selection.py's matching fix, same date.
        """
        count = self.num_tips
        # 1. Initial checks
        if not hasattr(dag, 'tips') or not dag.tips:
            return [dag.genesis['hash']] * count

        current_tips = list(dag.tips)

        # If there are fewer tips than requested, return all of them
        if len(current_tips) <= count:
            return current_tips

        # 2. Prepare data for random walk
        approvers: Dict[str, List[str]] = {}
        for tx_hash, tx_data in dag.vertices.items():
            for parent in tx_data.get('parents', []):
                if parent not in approvers:
                    approvers[parent] = []
                approvers[parent].append(tx_hash)

        # Clear cache for weights
        self.weights_cache = {}

        # 3. Perform Random Walk k times
        selected_tips: Set[str] = set()
        max_walk_attempts = count * 5  # Increase attempts if needed
        attempts = 0
        
        while len(selected_tips) < count and attempts < max_walk_attempts:
            attempts += 1
            
            # Start from the genesis
            curr = dag.genesis['hash']
            
            # Move towards the tips
            while curr not in current_tips:
                children = approvers.get(curr, [])
                
                # If we reach a leaf node (not a tip), go back to the genesis
                if not children:
                    break 
                
                # Calculate weights for children
                weights = []
                valid_children = []
                
                for child in children:
                    # Calculate cumulative weight
                    w = self._calculate_cumulative_weight(child, approvers, set())
                    # MCMC formula: exp(-alpha * (cumulative_weight))
                    # Or a simplified version: w ^ alpha
                    p_w = w ** self.alpha  # Higher weight means higher probability
                    weights.append(p_w)
                    valid_children.append(child)
                
                if not valid_children:
                    break

                # Select the next node based on weights
                curr = random.choices(valid_children, weights=weights, k=1)[0]
            
            # If we reach a tip, add it to the selected list
            if curr in current_tips:
                selected_tips.add(curr)

        # 4. Complete the list if not enough tips are found
        final_list = list(selected_tips)
        if len(final_list) < count:
            remaining = [t for t in current_tips if t not in final_list]
            needed = count - len(final_list)
            if remaining:
                final_list.extend(random.sample(remaining, min(len(remaining), needed)))
            else:
                # Fallback to genesis if not enough tips are available
                while len(final_list) < count:
                    final_list.append(dag.genesis['hash'])

        return final_list[:count]
