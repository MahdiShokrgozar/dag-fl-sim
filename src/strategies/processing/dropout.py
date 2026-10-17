# src/strategies/processing/dropout.py
import logging
from typing import Dict, Any, Optional
import torch
from src.strategies.processing.base import UpdateProcessor
from src.strategies.reputation.dbfl import DBFLReputation
from src.utils.dropout import apply_federated_dropout

logger = logging.getLogger(__name__)


def _cosine_similarity(w_after: Optional[Dict[str, Any]], w_before: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    sim(omega_m, omega_m^{e,0}) - Eq. (7): cosine similarity between this ES's
    trained model and its own pre-training model for this round, treating the
    whole state_dict as one flattened vector. Returns None (caller falls back
    to the paper's own "not yet available" default) if either side is
    missing or has no floating-point parameters in common - e.g. the very
    first round a node ever publishes has no `base_weights` to compare against.
    """
    if not w_after or not w_before:
        return None
    parts_after, parts_before = [], []
    for key, v_after in w_after.items():
        v_before = w_before.get(key)
        if (
            torch.is_tensor(v_after) and torch.is_tensor(v_before)
            and v_after.is_floating_point() and v_before.is_floating_point()
            and v_after.shape == v_before.shape
        ):
            parts_after.append(v_after.detach().float().flatten())
            parts_before.append(v_before.detach().float().flatten())
    if not parts_after:
        return None
    flat_after = torch.cat(parts_after)
    flat_before = torch.cat(parts_before)
    denom = flat_after.norm() * flat_before.norm()
    if denom.item() == 0.0:
        return None
    return float(torch.dot(flat_after, flat_before) / denom)

class ReputationBasedDropout(UpdateProcessor):
    """
    Implements Reputation-based Federated Dropout (DB-FL Paper, Section 2.5).
    
    The Edge Server (ES) generates different binary masks based on its reputation.
    - High Reputation -> Higher Dropout Rate (Forces robust learning, prevents overfitting)
    - Low Reputation -> Lower Dropout Rate
    """
    def __init__(self, config: Dict[str, Any], reputation_module: DBFLReputation):
        super().__init__(config)
        self.config = config
        self.reputation = reputation_module
        
        # Fetch dropout bounds from config dynamically
        dropout_config = config.get("protocols", {}).get("dbfl", {}).get("dropout", {})
        self.min_rate = float(dropout_config.get("min_rate", 0.0))
        self.max_rate = float(dropout_config.get("max_rate", 0.75)) # Up to 75% as tested in paper Fig 7
        self.enabled = bool(dropout_config.get("enabled", True))

        # Cumulative count of completed local-training rounds per ES - a
        # stand-in for "sum of local training rounds of IDs under the ES
        # jurisdiction" (indicator 2, Sec. 2.2). This project doesn't model a
        # separate ID tier under each ES, so one call to process_outbound
        # (one completed round for this node) is the closest available unit.
        self._round_counts: Dict[str, int] = {}

    def process_outbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        """
        Intercepts the outgoing global model and applies the reputation-based dropout mask
        to generate a sub-model for the targeted Intelligent Devices (IDs).
        """
        if not self.enabled:
            return payload

        weights = payload.get("weights")
        author = payload.get("author")
        
        if not weights or not author:
            return payload
            
        # 1. Determine Dropout Rate based on Reputation
        score = self.reputation.get_score(author)
        
        # Linearly map reputation score (0 - 100) to dropout rate (min_rate to max_rate)
        reputation_ratio = min(max(score / 100.0, 0.0), 1.0)
        dropout_rate = self.min_rate + (reputation_ratio * (self.max_rate - self.min_rate))
        
        logger.debug(f"Applying FedDropout for {author} with rate {dropout_rate:.2f} (Rep Score: {score:.2f})")

        # 2. Apply Mask (Dropout) using the robust utility function
        masked_weights = apply_federated_dropout(weights, dropout_rate)

        payload["weights"] = masked_weights

        # 3. Update this ES's reputation for the NEXT round (Algorithm 1,
        # line 16: the mask for round e used R^{e-1} above; this round's own
        # outcome only takes effect starting next round). Without this call
        # DBFLReputation.update() was never invoked anywhere in the codebase
        # - every node's score stayed frozen at initial_score forever, which
        # silently collapsed RTSA and this dropout rate to their
        # reputation-blind/fixed-rate baselines. See
        # thesis-hub/docs/decisions.md, 2026-09-15.
        self._round_counts[author] = self._round_counts.get(author, 0) + 1
        raw_similarity = _cosine_similarity(weights, payload.get("base_weights"))
        samples = None
        train_loader = getattr(node, "train_loader", None) if node is not None else None
        dataset_data = getattr(train_loader, "dataset_data", None)
        if dataset_data is not None:
            samples = len(dataset_data)
        metric_data = {"rounds": float(self._round_counts[author])}
        if samples is not None:
            metric_data["samples"] = float(samples)
        if raw_similarity is not None:
            # Sec. 2.2, right after Eq. (7): "the greater the cosine
            # similarity, the smaller the difference. Therefore, the
            # negation is required before normalizing" - _recalculate_
            # entropy_scores treats every indicator as higher-is-better
            # (Eq. 6), so the RAW similarity must be negated here, not
            # passed through. Confirmed the hard way on the first real run
            # (0j_dbfl_sanity_check, 2026-09-15): the lazy attacker (whose
            # weights barely change round to round, so raw similarity to its
            # own pre-round weights is near-maximal) landed the HIGHEST
            # reputation of any node in the network (95.4 vs honest nodes'
            # 6-38) - this indicator was rewarding the exact behavior PoL-
            # style defenses exist to catch. See
            # thesis-hub/docs/decisions.md, 2026-09-15.
            metric_data["similarity"] = -raw_similarity
        # node.last_accuracy, not payload["accuracy"]: base_node.py sets the
        # former unconditionally from the trainer's real return value right
        # after training, while the latter is the node's self-reported claim
        # - a malicious trainer can spoof_metadata() to lie about it (see
        # src/nodes/malicious/poisoning.py). Same class of "trust the claim,
        # not the ground truth" leak already found and fixed in IronForge's
        # PoL today; feeding a liar's own claim into ITS OWN reputation score
        # would make reputation self-defeating by construction.
        true_accuracy = getattr(node, "last_accuracy", None) if node is not None else None
        if true_accuracy is not None:
            metric_data["accuracy"] = float(true_accuracy)
        self.reputation.update(author, metric_data)

        # Attach metadata for metric collectors and tracking
        if "metadata" not in payload:
            payload["metadata"] = {}
        payload["metadata"]["dropout_rate"] = float(dropout_rate)
        
        return payload

    def process_inbound(self, payload: Dict[str, Any], node: Any = None) -> Dict[str, Any]:
        """
        Inbound models are aggregated as usual. Dropout is only applied when
        broadcasting the global model to form sub-models.
        """
        return payload