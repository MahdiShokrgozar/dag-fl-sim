# src/aggregation/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch

class BaseAggregator(ABC):
    """
    Abstract Base Class for all Aggregation Strategies.
    Defines the standard interface for client updates and server aggregation.
    """
    agg_type = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def client_update(self, node, global_weights: Optional[Dict] = None) -> Dict:
        """
        Executes the local training process on a client node.
        
        Args:
            node: The node instance performing the training.
            global_weights: Optional weights to initialize the model (e.g., from parent).
            
        Returns:
            Dict: A payload containing the updated weights and metadata.
        """
        pass

    def process_incoming_update(self, node, update_payload: Dict) -> Optional[Dict]:
        """
        Optional hook to process data before adding it to the queue.
        Useful for decompression, decryption, or downloading from IPFS.
        
        Args:
            node: The node receiving the data.
            update_payload: The raw dictionary received from the network.
            
        Returns:
            Dict: The processed payload ready for aggregation, or None to discard.
        """
        return update_payload

    @abstractmethod
    def aggregate(self, payloads: List[Dict], node: Any = None) -> Dict:
        """
        Aggregates a list of model updates into a single weight dictionary.

        Args:
            payloads: List of update dictionaries received from peers.
            node: The aggregating node, for strategies that need its local view
                (anchor selection, per-node trust weighting). Optional -
                strategies that do not need it should accept and ignore it.

        Returns:
            Dict: The aggregated model weights.

        Note:
            `node` is part of the interface because BaseNode has always called
            `aggregate(payloads, node=self)`. Only the two FARABI aggregators
            declared it, so every other strategy - fedavg, fedprox, median,
            trimmed_mean, krum, weighted_fedavg, ironforge - raised TypeError
            the moment it was asked to aggregate. IronForge only surfaced this
            on 2026-09-12 because it used to die earlier, at the unguarded
            reputation check. See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        pass