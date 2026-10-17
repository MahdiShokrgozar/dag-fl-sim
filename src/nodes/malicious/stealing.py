import copy
import logging
import torch
from typing import Dict, Any, Tuple

from src.nodes.base_node import BaseNode
from src.nodes.malicious.base_malicious import MaliciousTrainer
from src.utils.ipfs_utils import download_model_from_ipfs, MockIPFS
from src.nodes.malicious.lazy import add_gaussian_noise

logger = logging.getLogger("Node.Stealing")

class StealingTrainer(MaliciousTrainer):
    """
    Plagiarizes the best public model from the DAG tips.
    """

    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        node = self.node
        node.logger.info(f"[{node.node_id}] Executing STEALING attack: hunting for a victim.")

        atk_cfg = node.config.get("participants", {}).get("attacks", {}).get("stealing", {})
        min_victim_accuracy = float(atk_cfg.get("min_victim_accuracy", 0.0))
        fake_compute_time = float(atk_cfg.get("fake_compute_time", 0.05))

        dag = getattr(node.simulator, "dag", None)
        victim_tx, victim_id, victim_acc = None, None, -1.0

        for tip_hash in list(getattr(dag, "tips", [])) if dag else []:
            tx = dag.vertices.get(tip_hash)
            if not tx: continue
            
            author = tx.get("author", tx.get("proposer", "unknown"))
            if author == node.node_id: continue
            
            claimed = float(tx.get("payload", {}).get("accuracy", 0.0))
            if claimed > victim_acc:
                victim_acc, victim_id, victim_tx = claimed, author, tx

        stolen_weights = None
        if victim_tx is not None and victim_acc >= min_victim_accuracy:
            cid = victim_tx.get("payload", {}).get("ipfs_cid")
            storage_config = getattr(node.processor, "storage_config", node.config.get("storage", {}))
            mock_ipfs = getattr(node.simulator, "mock_ipfs", MockIPFS())
            
            try:
                data = download_model_from_ipfs(cid, storage_config, mock_ipfs)
                stolen_weights = copy.deepcopy(data[0]) if data else None
            except Exception as e:
                logger.error(f"[{node.node_id}] Steal download failed: {e}")

        if stolen_weights:
            # Noisy stealing (thesis-hub/docs/decisions.md, 2026-09-30, in-band
            # attackers): perturb the copy so its content hash (CID) no longer
            # matches the victim's. noise_std is per parameter; default 0.0 draws
            # no random numbers, so the original exact-copy attack is unchanged.
            noise_std = float(atk_cfg.get("noise_std", 0.0))
            if noise_std > 0.0:
                stolen_weights = add_gaussian_noise(stolen_weights, noise_std)
            # We must evaluate to get true accuracy for the Oracle
            temp_model = self.node.cache_manager.checkout_model(stolen_weights)
            true_acc, _ = self.evaluate_state(temp_model)
            self.node.cache_manager.checkin_model(temp_model)
            
            node.logger.warning(f"💉 [{node.node_id}] STOLE model from {victim_id} (true: {true_acc:.2f}%)")
            
            # Spoof to claim slightly better or equal to the victim
            self.spoof_metadata(target_accuracy=victim_acc + 0.5)
            return stolen_weights, fake_compute_time, true_acc

        node.logger.info(f"[{node.node_id}] No valid victim; falling back to lazy.")
        self.spoof_metadata(target_accuracy=90.0)
        return copy.deepcopy(init_weights), fake_compute_time, getattr(self.node, 'last_accuracy', 0.0)

class StealingNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)         
        self.trainer = StealingTrainer(self)