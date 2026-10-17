"""
Module: farabi_processor.py
Description: 
    Handles the preparation of outbound federated learning updates.
    Implements a Hybrid Payload mechanism to optimize bandwidth and 
    injects Phase-Aware signaling (Exploration/Exploitation) into the block metadata.
"""

import logging
import torch
from typing import Dict, Any

from src.utils.ipfs_utils import upload_model_to_ipfs, MockIPFS
from src.strategies.processing.base import UpdateProcessor

logger = logging.getLogger(__name__)

class FarabiUpdateProcessor(UpdateProcessor):
    """
    Processes raw local training outputs into lightweight DAG transactions.
    Extracts Delta Weights or Full Weights depending on the genesis phase.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the processor with global configurations.

        Args:
            config (Dict[str, Any]): Global simulation configuration dictionary.
        """
        super().__init__(config)
        self.node = None 
        self.storage_config = self.config.get("storage", {"ipfs_mock_enabled": True})
        
        consensus_cfg = self.config.get("protocols", {}).get("farabi", {}).get("consensus", {})
        self.grace_period_time = consensus_cfg.get("grace_period_time", 30.0)
        agg_cfg = self.config.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.weighting_mode = agg_cfg.get("weighting", "accuracy")  # "accuracy" یا "uniform"
        
    def _generate_lightweight_fingerprint(self, weights: Dict[str, torch.Tensor]) -> float:
        """
        Generates an O(1) mathematical fingerprint from the classifier head.

        Args:
            weights (Dict[str, torch.Tensor]): The PyTorch state_dict to hash.

        Returns:
            float: The numeric fingerprint for integrity verification.
        """
        if not weights: return 0.0
        try:
            last_tensor = weights[list(weights.keys())[-1]]
            return float(torch.sum(torch.abs(last_tensor)).item())
        except Exception:
            return 0.0

    def process_outbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        """
        Intercepts outbound payloads, applies IPFS offloading, and injects 
        critical cryptographic and game-theoretic metadata.

        Args:
            payload (Dict[str, Any]): Raw training data output.
            node (Any, optional): Reference to the local node instance.

        Returns:
            Dict[str, Any]: Formatted and lightweight payload ready for gossip.
        """
        self.node = node 
        parent_hashes = payload.get("parents", [])
        parent_accuracies = payload.get("parent_accuracies", [])
        
        logger.debug(f"🔍 [X-RAY 2.1 - Processor IN] Author: {payload.get('author', 'unknown')} | Payload Keys: {list(payload.keys())} | Hashes: {parent_hashes} | Accuracies: {parent_accuracies}")
        
        processed_payload = {
            "author": payload.get("author", "unknown"),
            "accuracy": payload.get("accuracy", 0.0),
            "parents": payload.get("parents", [])
        }

        # Declare the set that was actually aggregated. It equals `parents` in
        # unified/anchored mode, but in dual mode the aggregator substituted
        # finalized anchors and recorded them on the node. A validator rebuilds
        # the base from this field (falling back to `parents` when absent), so
        # without it every honest dual-mode block would fail its own consistency
        # check against its claimed weights. Injected here, in the protocol's own
        # payload formatter, so the generic node never has to know about it.
        agg_refs = getattr(node, "farabi_aggregation_refs", None) if node else None
        processed_payload["aggregation_refs"] = (
            list(agg_refs) if agg_refs else list(processed_payload["parents"])
        )
        # Diagnostic only (2026-09-10, thesis-hub/docs/decisions.md same date):
        # companion to the DIAG-DISTANCE log in farabi_consensus.py - lets a
        # proposer's OWN declared aggregation_refs (what it actually trained
        # from) be compared against what a later validator rebuilds from the
        # same field, for the same block. Remove once the dual-mode lazy-gate
        # investigation concludes.
        logger.debug(
            f"[DIAG-PROPOSE] author={processed_payload['author']} "
            f"parents_n={len(processed_payload['parents'])} "
            f"aggregation_refs_n={len(processed_payload['aggregation_refs'])} "
            f"used_farabi_aggregation_refs={bool(agg_refs)}"
        )

        parent_given_weights = {}
        
        if self.weighting_mode == "accuracy" and parent_hashes and parent_accuracies:
            # وزن متناسب با دقت والدین
            total_parent_acc = sum(parent_accuracies)
            for p_hash, p_acc in zip(parent_hashes, parent_accuracies):
                parent_given_weights[p_hash] = (p_acc / total_parent_acc) if total_parent_acc > 0 else (1.0 / len(parent_hashes))
            logger.info(f"✅ [PROCESSOR] Accuracy-weighted: {parent_given_weights}")
            
        elif parent_hashes:
            # وزن یکنواخت (fallback یا uniform mode)
            w = 1.0 / len(parent_hashes)
            for p in parent_hashes:
                parent_given_weights[p] = w
            if self.weighting_mode == "uniform":
                logger.info(f"⚖️ [PROCESSOR] Uniform-weighted: {parent_given_weights}")
            else:
                logger.warning(f"⚠️ [PROCESSOR] Fallback to uniform (missing parent_accuracies): {parent_given_weights}")
        else:
            logger.debug(f"[PROCESSOR] No parents to weight (genesis or empty block)")

                    
        processed_payload["parent_accuracies"] = parent_accuracies                  
        processed_payload["given_approval_weights"] = parent_given_weights
        
        processed_payload["version"] = self.config.get("protocols", {}).get("farabi", {}).get("block_version", "farabi_v4")
        processed_payload["class_accuracies"] = getattr(node, 'per_class_accuracy', payload.get("class_accuracies", {}))
        
        # 💡 Retrieve phase state injected by FarabiTipSelector
        current_mode = getattr(node, 'current_selection_mode', "exploitation")
        processed_payload["selection_mode"] = current_mode
            
        current_time = getattr(node.simulator, 'current_time', 0.0) if node and hasattr(node, 'simulator') else 0.0
        block_id = payload.get("block_id", payload.get("hash", f"blk_{processed_payload['author']}_{current_time}"))
        processed_payload["block_id"] = block_id
        processed_payload["timestamp"] = payload.get("timestamp", current_time)

        new_weights = payload.get("weights", {})
        base_weights = payload.get("base_weights", {})
        
        if new_weights:
            processed_payload["model_fingerprint"] = self._generate_lightweight_fingerprint(new_weights)
            
            is_genesis = (not base_weights) or (current_time < self.grace_period_time)
            is_full_model = True  # TODO: Fix Async State Corruption later
            processed_payload["is_full_model"] = True 
            
            # Delta logic preserved for future research; currently disabled.
            # To re-enable: set data_to_upload = delta and is_full_model = is_genesis
            # if is_genesis:
            #     data_to_upload = new_weights
            # else:
            #     data_to_upload = {k: new_weights[k] - base_weights[k] for k in new_weights.keys() if k in base_weights}
            
            data_to_upload = new_weights  
            total_params = sum(t.numel() for t in data_to_upload.values())
            size_bytes = total_params * 4
            
            mock_ipfs = getattr(node.simulator, 'mock_ipfs', MockIPFS()) if node and hasattr(node, 'simulator') else None
            cid = upload_model_to_ipfs(data_to_upload, self.storage_config, mock_ipfs)
            processed_payload["ipfs_cid"] = cid
            
            if node and hasattr(node, 'telemetry') and hasattr(node, 'calculate_upload_delay'):
                ipfs_upload_delay = node.calculate_upload_delay(size_bytes)
                node.telemetry.register_network_delay(ipfs_upload_delay)
                node.telemetry.register_traffic(size_bytes, channel="ipfs")
                node.telemetry.register_compute(0.005) 
        else:
            processed_payload["ipfs_cid"] = None
            processed_payload["model_fingerprint"] = 0.0

        return processed_payload

    def process_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Processes incoming data before it hits the message queue."""
        return payload