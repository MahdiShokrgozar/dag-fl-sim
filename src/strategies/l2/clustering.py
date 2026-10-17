# src/strategies/l2/clustering.py
import random
from typing import Dict, Any, List
from .base import L2Strategy

class RandomClustering(L2Strategy):
    """
    Standard Random Clustering Implementation.
    Groups nodes into clusters. One 'Head' aggregates, others are 'Workers'.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cluster_size = config.get("strategy", {}).get("cluster_size", 5)

    # --- 1. Formation (Topology & Roles) ---
    def form_l2_topology(self, nodes: Dict[str, Any], current_time: float) -> None:
        """
        Assigns roles (Head/Worker) based on random shuffling.
        Matches the abstract method signature in L2Strategy.
        """
        node_list = list(nodes.values())
        random.shuffle(node_list)

        cluster_count = 0
        for i in range(0, len(node_list), self.cluster_size):
            chunk = node_list[i : i + self.cluster_size]
            if not chunk: continue
            
            cluster_id = f"cls_{cluster_count}_{int(current_time)}"
            head_node = chunk[0] # First node is Head
            
            for node in chunk:
                role = "head" if node == head_node else "worker"
                
                # Update Node State (Critical for BaseNode permission check)
                node.l2_state.update({
                    "active_session": cluster_id,
                    "role": role,
                    "head_id": head_node.node_id,
                    "peers": [n.node_id for n in chunk if n != node]
                })
            
            cluster_count += 1

    # --- 2. Intra-Layer Operation ---
    def select_aggregator(self, cluster_members: List[str], l2_state: Dict) -> str:
        """
        The Cluster Head is always the aggregator in Random Clustering.
        """
        return l2_state.get("head_id")

    def validate_l2_update(self, update: Dict, l2_state: Dict) -> bool:
        """
        Simple validation: Accept all updates from cluster peers.
        In a real protocol, we would check signatures here.
        """
        return True

    # --- 3. Settlement (Interaction with L1) ---
    def should_commit_to_l1(self, l2_state: Dict, current_time: float) -> bool:
        """
        Only the 'head' is allowed to commit blocks to the main DAG.
        Workers send to Head (simulated), Head sends to L1.
        """
        return l2_state.get("role") == "head"

    def create_commitment_payload(self, aggregated_model: Any, l2_metadata: Dict) -> Dict[str, Any]:
        """
        Wraps the aggregated model into a format ready for L1.
        """
        return {
            "weights": aggregated_model,
            "meta": l2_metadata,
            "type": "L2_COMMITMENT"
        }