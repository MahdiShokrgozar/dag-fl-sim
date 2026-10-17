from typing import Dict,List, Any

from .base import TipSelector

class LatestTipSelector(TipSelector):
    """
    Selects the most recently added tips based on timestamp/order.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_tips = int(config.get("protocol", {}).get("num_tips", 5))

    def select_tips(self, dag: Any, node: Any = None) -> List[str]:
        # BaseNode.create_and_broadcast_block calls select_tips(dag, node=self)
        # - this used to take a `count` kwarg instead, so every call from the
        # live simulator raised TypeError: unexpected keyword argument
        # 'node'. See random_selection.py's matching fix, same date.
        # Assume that tips in DAG are stored as {hash: metadata}
        # And metadata contains timestamp

        tip_hashes = list(dag.tips)

        tips_objects = []
        for tx_hash in tip_hashes:
            if tx_hash in dag.vertices:
                block = dag.vertices[tx_hash]
                tips_objects.append((tx_hash, block.get('timestamp', 0)))

        sorted_tips = sorted(tips_objects, key=lambda x: x[1], reverse=True)

        # Select the k newest tips
        selected_hashes = [item[0] for item in sorted_tips[:self.num_tips]]

        if not selected_hashes:
            return tip_hashes

        return selected_hashes
