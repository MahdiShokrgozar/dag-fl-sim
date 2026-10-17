# core/dag.py
import logging
import random
from typing import List,Optional, Any
import networkx as nx
import matplotlib.pyplot as plt
import os

class DAG:
    def __init__(self):
        self.vertices = {}
        self.tips = set()
        self.genesis = self._create_genesis()
        # Add genesis to vertices immediately upon creation
        self.vertices[self.genesis['hash']] = self.genesis
        self.tips.add(self.genesis['hash'])
        self.last_settlement_hash = self.genesis['hash']


    def _create_genesis(self) -> dict:
        """Initialize DAG with valid genesis transaction"""
        genesis = {
            'hash': '0'*64,
            'parents': [],
            'creator': 'system',
            'timestamp': 0, # Use logical clock 0 for Genesis
            'update': {'metadata': 'GENESIS'}
        }
        return genesis

    def add_transaction(self, tx: dict) -> bool: 
        """Add validated transaction to DAG"""
        
        tx_hash = tx['hash']

        if tx_hash in self.vertices:
            logging.debug(f"Transaction rejected: hash {tx_hash[:8]} already exists.")
            return False

        self.vertices[tx_hash] = tx
        self.tips.add(tx_hash)

        if tx.get("update", {}).get("type") == "EpochSettlement":
            self.last_settlement_hash = tx_hash

        for parent in tx['parents']:
            if parent in self.tips:
                self.tips.remove(parent)

        return True

    def validate(self, tx: dict, verbose: bool = False) -> bool:
        """Validate a transaction before adding it to the DAG."""
        if tx['hash'] in self.vertices:
            if verbose: logging.debug(f"Validation failed: hash {tx['hash'][:8]} already exists.")
            return False
        
        # Bypass validation for settlement TX
        if tx.get("update", {}).get("type") == "EpochSettlement":
            return True
    
        if not self.vertices:  
            return True
        
        
        # Ensure all parents exist in the DAG (except Genesis hash '0'*64 which is implicit)
        for p_hash in tx['parents']:
            if p_hash == '0'*64:
                continue

            # If parent missing → try to sync from main DAG (global L1)
            if p_hash not in self.vertices:
                if hasattr(self, "sync_callback") and self.sync_callback:
                    parent_tx = self.sync_callback(p_hash)
                    if parent_tx:
                        self.add_transaction(parent_tx)
                    else:
                        if verbose: logging.debug(f"Validation failed: missing parent {p_hash[:8]} not found anywhere.")
                        return False
                else:
                    return False
                
        return True


    def get_tips(self, k: Optional[int] = None) -> List[str]:
        """
        Return up to *k* tip hashes (or all of them if k is None / bigger
        than tip count).
        """
        tips_list = list(self.tips)
        if k is None or k >= len(tips_list):
            return tips_list
        # Ensure k is not greater than the number of available tips to avoid ValueError in random.sample
        return random.sample(tips_list, k=min(k, len(tips_list)))

    def visualize_dag(self, path: str = "output/dag_full.png"):
        """Draw *every* vertex, colour by round, tips with a red outline."""
        g = nx.DiGraph()
        rounds = {} # hash -> logical round (depth)

        for v_hash, v_data in self.vertices.items(): # Iterate with hash and data
            g.add_node(v_hash) # Use hash as node ID in networkx graph
            if not v_data["parents"]: # Genesis or disconnected
                rounds[v_hash] = 0
            else:
                parent_rounds = [rounds[p] for p in v_data["parents"] if p in rounds]
                if parent_rounds:
                    rounds[v_hash] = 1 + max(parent_rounds)
                else: # Fallback if parents not found in rounds (e.g., partial DAG data)
                    rounds[v_hash] = 0
            for p_hash in v_data["parents"]: # Iterate through parent hashes
                if p_hash in self.vertices: # Ensure parent exists in DAG before adding edge
                    g.add_edge(p_hash, v_hash) # Edge from parent to child

        if not rounds: # Handle empty DAG (only Genesis, no rounds computed)
            logging.warning("No rounds computed for DAG visualization. Skipping coloring.")
            node_colors = "blue"
            max_round_val = 0
        else:
            max_round_val = max(rounds.values())
            norm_factor = (max_round_val + 1) if max_round_val > 0 else 1
            # Ensure nodes are in 'g' for coloring. 'n' is the node ID (hash) in graph 'g'
            node_colors = [plt.cm.plasma(rounds.get(n, 0) / norm_factor) for n in g.nodes()] # Use g.nodes()

        pos = nx.spring_layout(g, k=0.4, seed=0) # k=0.4 for spacing, seed for reproducibility

        nx.draw(g, pos,
                node_size=120,
                node_color=node_colors,
                edge_color="#444444",
                arrowsize=6,
                with_labels=False)

        # Highlight current tips
        nx.draw_networkx_nodes(g, pos,
                nodelist=list(self.tips), # Use self.tips
                node_size=160,
                node_color="none",
                edgecolors="red",
                linewidths=1.8)

        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure path's directory exists
        plt.title(f"DAG after round {max_round_val} • {len(g)} vertices")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def to_dot_with_labels(self, path: str = "dag_with_labels.dot"):
        """
        Outputs graph in DOT format with full labels on each node.
        Horizontal graph structure (rankdir=RL) and displays hash and metadata.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure path's directory exists

        with open(path, "w") as f:
            f.write("digraph DAG {\n")
            f.write("rankdir=RL;\n") # Horizontal direction from right to left (RL = Right to Left)
                                       # If you want LR (Left to Right) change to LR
            f.write('node [shape=box, style=filled, fillcolor=lightyellow, fontname="Courier"];\n')

            for v_hash, v_data in self.vertices.items(): # Iterate with hash and data
                label_lines = [f"hash: {v_hash[:8]}"] # Display truncated hash
                # Access _original_creator_id from update if present
                if isinstance(v_data.get("update"), dict) and "_original_creator_id" in v_data["update"]:
                    label_lines.append(f"creator: {v_data['update']['_original_creator_id']}")
                elif v_data.get("creator"): # Fallback to original creator if _original_creator_id not in update
                    label_lines.append(f"creator: {v_data['creator']}")

                label_lines.append(f"time: {v_data['timestamp']}") # Display timestamp

                label = "\\n".join(label_lines).replace('"', '\\"') # Format label for DOT
                f.write(f'"{v_hash}" [label="{label}"];\n') # Use v_hash for node ID

            for v_hash, v_data in self.vertices.items():
                for p_hash in v_data["parents"]:
                    # Arrow from child (v_hash) to parent (p_hash)
                    f.write(f'"{v_hash}" -> "{p_hash}";\n')

            f.write("}\n")

    def get_latest_settlement_tx_hash(self):
        return self.last_settlement_hash
