# src/strategies/selection/random_selection.py
import random
from typing import Dict, List, Any
from .base import TipSelector

class RandomTipSelector(TipSelector):
    """
    Selects 'k' random tips from the DAG's available tips.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_tips = int(config.get("protocol", {}).get("num_tips", 5))

    def select_tips(self, dag: Any, node: Any = None) -> List[str]:
        # BaseNode.create_and_broadcast_block calls select_tips(dag, node=self)
        # (see TipSelector.select_tips's abstract signature) - this used to
        # take a `count` kwarg instead, so every call from the live
        # simulator raised TypeError: unexpected keyword argument 'node'.
        # GenericProtocol (the only protocol that wires this selector in)
        # could never actually run a round. Same bug existed in
        # latest_selection.py and mcmc_selection.py. See
        # thesis-hub/docs/decisions.md, 2026-09-15.
        available_tips = list(dag.tips)

        # If fewer tips than requested, return all of them
        if len(available_tips) <= self.num_tips:
            return available_tips

        # Select k random tips
        return random.sample(available_tips, self.num_tips)