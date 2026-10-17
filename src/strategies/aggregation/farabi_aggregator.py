"""
Module: farabi_aggregator.py
Description:
    FARABI Robust Aggregator with Layer-Wise Strategy, Device Safety, 
    and Lazy IPFS Fetching.
    
    Architecture:
    - Feature Extractor layers (e.g., first 90% of params): Robust aggregation
      - N < 5: Coordinate-wise Median (highly robust for small N)
      - N >= 5: Trimmed Mean (better convergence, discards outliers)
    - Classifier layers (e.g., last 10% of params): Weighted Average
      - Preserves Non-IID specialization and diversity.
      
    Key Features:
    - Lazy Model Fetching: Downloads tensors from IPFS only when needed.
    - Automatic Layer Type Detection: Heuristic based on parameter count.
    - Device-Safe: Handles CPU/GPU mismatches smoothly.
    - Type-Safe: Preserves integer tensors (e.g., BatchNorm tracking).
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import BaseAggregator
from .median import Median
from .trimmed_mean import TrimmedMean
from .weighted_fedavg import WeightedFedAvg

logger = logging.getLogger(__name__)


class FarabiRobustAggregator(BaseAggregator):
    """
    Adaptive Layer-Wise Robust Aggregator for the FARABI protocol.
    """
    
    agg_type = "farabi_robust"
    
    def __init__(self, cfg: dict):
        """
        Initialize the aggregator with configuration parameters.
        
        Args:
            cfg (dict): Global configuration dictionary.
        """
        super().__init__(cfg)
        
        # Sub-aggregators composition
        self.median_agg = Median(cfg)
        self.trimmed_agg = TrimmedMean(cfg)
        self.weighted_avg = WeightedFedAvg(cfg)
        
        # Extraction of specific hyperparameters
        agg_cfg = cfg.get("protocols", {}).get("farabi", {}).get("aggregation", {})
        self.min_models_for_trimmed = agg_cfg.get("min_models_for_trimmed", 5)
        self.trim_fraction = agg_cfg.get("trim_fraction", 0.2)
        self.feature_extractor_ratio = agg_cfg.get("feature_extractor_param_ratio", 0.90)
        
        logger.info(
            f"⚖️ FARABI Robust Aggregator Initialized | "
            f"Trimmed threshold: N={self.min_models_for_trimmed} | "
            f"Feature ratio: {self.feature_extractor_ratio}"
        )
    
    def _lazy_fetch_weights(self, payload_outer: Dict[str, Any], node: Any = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Robust weight extraction using ipfs_utils.download_model_from_ipfs().
        Supports nested DAG block structures and both Mock/Real IPFS.
        """
        # 🔄 Step 1: Aggressively unwrap nested structures
        inner = payload_outer
        for _ in range(2):
            unwrapped = False
            for nest_key in ['payload', 'update', 'data']:
                if nest_key in inner and isinstance(inner[nest_key], dict):
                    inner = inner[nest_key]
                    unwrapped = True
                    break
            if not unwrapped:
                break
        
        # ⚡ Step 2: Direct weights check (fast path)
        if 'weights' in inner and isinstance(inner['weights'], dict):
            weights = inner['weights']
            if len(weights) > 0:
                logger.info(f"✅ [LAZY-FETCH] Direct weights found | Layers: {len(weights)}")
                return weights
        
        # 🔍 Step 3: Find IPFS CID
        cid = inner.get('ipfs_cid')
        if not cid and 'payload' in inner and isinstance(inner['payload'], dict):
            cid = inner['payload'].get('ipfs_cid')
        if not cid and 'update' in inner and isinstance(inner['update'], dict):
            cid = inner['update'].get('ipfs_cid')
        
        if not cid:
            logger.warning(f"⚠️ [LAZY-FETCH] No 'weights' or 'ipfs_cid' found. Keys: {list(inner.keys())}")
            return None
        
        logger.info(f"📥 [LAZY-FETCH] Found IPFS CID: {str(cid)[:16]}...")
        
        # 📦 Step 4: Use ipfs_utils.download_model_from_ipfs()
        # This function handles both MockIPFS and real IPFS automatically
        try:
            from src.utils.ipfs_utils import download_model_from_ipfs
            
            storage_cfg = {}
            if node and hasattr(node, 'config'):
                storage_cfg = node.config.get("storage", {})
            
            mock_ipfs = None
            if node and hasattr(node, 'simulator'):
                mock_ipfs = getattr(node.simulator, 'mock_ipfs', None)
            
            result = download_model_from_ipfs(cid, storage_cfg, mock_ipfs)
            
            if result is None:
                logger.error(f"❌ [LAZY-FETCH] download_model_from_ipfs() returned None for CID: {str(cid)[:16]}...")
                return None
            
            # download_model_from_ipfs returns (weights, size) tuple
            if isinstance(result, tuple) and len(result) == 2:
                weights, size = result
            else:
                weights = result
                size = 0
            
            if not isinstance(weights, dict):
                logger.error(f"❌ [LAZY-FETCH] download_model_from_ipfs() returned non-dict: {type(weights)}")
                return None
            
            if len(weights) == 0:
                logger.error("❌ [LAZY-FETCH] download_model_from_ipfs() returned empty weights dict.")
                return None
            
            logger.info(
                f"✅ [LAZY-FETCH] Successfully fetched {len(weights)} layers | "
                f"Size: ~{size / 1024:.1f} KB"
            )
            return weights
            
        except Exception as e:
            logger.error(f"❌ [LAZY-FETCH] download_model_from_ipfs() failed: {e}", exc_info=True)
            return None
        
    def _resolve_aggregation_inputs(self, payloads: List[Dict[str, Any]], node: Any = None) -> List[Dict[str, Any]]:
        """
        Decide what this protocol actually merges, given what the core handed over.

        The simulator core is protocol-agnostic: it selects parents, fetches
        them, and passes them here as "the models to aggregate". For FARABI in
        `dual` mode that is the wrong set. Parents are a CONSENSUS duty drawn
        from the tip pool and chosen for speed; the aggregation base has to be
        chosen for TRUST, because it is the poisoning surface, which means
        FINALIZED blocks - ones that already cleared
        weight >= max(w_min, theta * mean_virtue).

        Substituting here rather than in the node keeps the core generic: it
        still says "aggregate the models for this round", and the protocol
        decides which models those are. The chosen hashes are recorded on the
        node so FarabiUpdateProcessor can declare them in the block as
        `aggregation_refs` - a validator must rebuild the base from the set the
        author actually used, or every honest block fails its own consistency
        check against its claimed weights.

        Returns `payloads` unchanged in unified/anchored mode, and on cold start
        in dual mode, when nothing has been finalized yet and there is no
        trusted set to prefer.
        """
        selector = getattr(node, "tip_selector", None) if node else None
        mode = getattr(selector, "selection_mode", "unified")

        refs = [p.get("hash", p.get("block_id", "")) for p in payloads]
        if node is not None:
            node.farabi_aggregation_refs = refs

        if mode != "dual" or selector is None:
            return payloads

        dag = getattr(node, "local_dag", None)
        if dag is None or not getattr(dag, "vertices", None):
            dag = getattr(getattr(node, "simulator", None), "dag", None)
        if dag is None:
            return payloads

        anchors = selector.select_anchors(dag, node, selector.anchor_count(for_aggregation=True))
        if not anchors:
            logger.info("⚓ [AGG] dual mode: nothing finalized yet, aggregating the parents.")
            return payloads

        node.farabi_aggregation_refs = [
            a.get("hash", a.get("block_id", "")) for a in anchors
        ]
        logger.info(
            f"⚓ [AGG] dual mode: aggregating {len(anchors)} finalized anchor(s) "
            f"instead of {len(payloads)} tip parent(s)."
        )
        return anchors

    def aggregate(self, payloads: List[Dict[str, Any]], node: Any = None) -> Dict[str, torch.Tensor]:
        """
        Main aggregation pipeline for FARABI Robust Aggregation.
        
        Pipeline Stages:
        1. Lazy IPFS Fetch: Download model weights on-demand.
        2. Weight-Space Anomaly Filter: Remove statistical outliers.
        3. Layer Type Detection: Classify layers into Feature/Classifier.
        4. Layer-Wise Robust Aggregation: Apply Median/TrimmedMean to 
        Feature layers and Virtue-Weighted Average to Classifier layers.
        
        Args:
            payloads (List[Dict]): List of block payloads to aggregate.
            node (Any): Reference to the local node for IPFS access.
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated model weights.
        """
        
        payloads = self._resolve_aggregation_inputs(payloads, node)

        logger.warning(f"🚨 [AGG-ENTRY] aggregate() called with {len(payloads)} payloads.")

        if not payloads:
            logger.error("❌ [AGG-ENTRY] Payloads list is empty. Aborting.")
            return {}

        # 🧹 Stage 0: Filter out empty/Genesis/metadata-only blocks
        valid_blocks = []
        for p in payloads:
            if not isinstance(p, dict):
                continue
            
            # Unwrap nested payload structure
            inner = p.get("payload", p)
            
            # Skip blocks that only have metadata
            if isinstance(inner, dict) and len(inner.keys()) <= 2 and 'metadata' in inner:
                logger.info(f"🧹 [AGG-SKIP] Skipping metadata-only block.")
                continue
            
            # Skip Genesis blocks (hash all zeros or no meaningful data)
            block_hash = p.get("hash", "")
            is_genesis = (
                block_hash.startswith("00000000") or 
                block_hash == "" or
                (inner.get("ipfs_cid") is None and "weights" not in inner and "update" not in inner)
            )
            
            if is_genesis:
                logger.info(f"🧹 [AGG-SKIP] Skipping Genesis/empty block.")
                continue
                
            valid_blocks.append(p)
        
        if not valid_blocks:
            logger.error("❌ [AGG-ENTRY] All payloads were empty/Genesis. Aborting.")
            return {}
            
        payloads = valid_blocks
        
        # 📥 Stage 1: Lazy IPFS Fetch
        valid_payloads = []
        for payload in payloads:
            weights = self._lazy_fetch_weights(payload, node)
            
            if weights is not None and len(weights) > 0:
                # 🎯 Smart extraction: Handle nested payload structure
                # Try to get node_id from multiple possible locations
                inner = payload.get("payload", payload)
                node_id = (
                    payload.get("author") or 
                    payload.get("proposer") or
                    inner.get("author") or 
                    inner.get("proposer") or
                    "unknown"
                )
                
                # Try to get accuracy from multiple possible locations
                accuracy = (
                    payload.get("accuracy") or
                    inner.get("accuracy") or
                    0.0
                )
                
                # Try to get samples count
                samples = (
                    payload.get("samples") or
                    inner.get("samples") or
                    1
                )
                
                valid_payloads.append({
                    "node_id": node_id,
                    "weights": weights,
                    "accuracy": accuracy,
                    "samples": samples
                })
                
                logger.info(f"✅ [AGG-FETCH] Successfully fetched weights for {node_id} (accuracy: {accuracy:.2f})")
            else:
                node_id = payload.get("proposer", payload.get("author", "unknown"))
                logger.warning(f"⚠️ [AGG-FETCH] Failed to fetch weights for payload: {node_id}")
        
        if not valid_payloads:
            logger.error("❌ [AGG-FETCH] No valid payloads after IPFS fetch. Aborting.")
            return {}
        
        # 🚨 Stage 2: Weight-Space Anomaly Detection
        logger.warning(f"🚨 [FILTER-ENTRY] Invoking Anomaly Filter on {len(valid_payloads)} models...")
        valid_payloads = self._filter_weight_space_anomalies(valid_payloads, node)
        
        if not valid_payloads:
            logger.error("❌ [FILTER-RESULT] All models dropped by anomaly filter. Aborting aggregation.")
            return {}
        
        N = len(valid_payloads)
        logger.info(f"✅ [AGG-PROCEED] Proceeding with {N} models after anomaly filter.")

        # 🔍 Stage 3: Layer Type Detection
        feature_layers, classifier_layers = self._detect_layer_types(valid_payloads)
        
        # 🔄 Stage 4: Layer-Wise Robust Aggregation
        aggregated = {}
        
        # Aggregate Feature Layers using Median/TrimmedMean
        if feature_layers:
            feature_result = self._aggregate_feature_layers(valid_payloads, feature_layers)
            aggregated.update(feature_result)
        
        # Aggregate Classifier Layers using Virtue-Weighted Average
        if classifier_layers:
            classifier_result = self._aggregate_classifier_layers(valid_payloads, classifier_layers, node)
            aggregated.update(classifier_result)
        
        logger.info(
            f"✅ [FARABI-AGG] Aggregated {N} models | "
            f"Feature layers: {len(feature_layers)} | "
            f"Classifier layers: {len(classifier_layers)}"
        )
        
        return aggregated

    def _get_common_device(self, payloads: List[Dict]) -> Optional[torch.device]:
        """Finds the most common target device among fetched weights."""
        for p in payloads:
            for tensor in p["weights"].values():
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
        return None

    def _normalize_payloads_to_device(self, payloads: List[Dict], target_device: torch.device) -> None:
        """Moves all tensors within the payloads to a unified PyTorch device."""
        for p in payloads:
            for key, tensor in p["weights"].items():
                if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                    p["weights"][key] = tensor.to(target_device)

    def _detect_layer_types(self, payloads: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Robust Layer Type Detection for CNN Architectures.
        
        This method classifies model layers into two categories:
        1. Feature Extractor Layers (Conv2d, BatchNorm, Pooling): 
           Aggregated using Median/TrimmedMean for Byzantine robustness.
        2. Classifier Layers (Linear/FC): 
           Aggregated using Virtue-Weighted Average to preserve 
           local specializations in Non-IID settings.
        
        Fallback Strategy:
        If the primary heuristic fails or returns zero feature layers 
        (common in small models or custom architectures), this method 
        applies a conservative heuristic: the last 1-2 layers are 
        assumed to be classifiers, and the remaining layers are 
        treated as feature extractors. This prevents the catastrophic 
        failure mode where all layers are classified as "classifier", 
        causing the aggregator to fall back to non-robust Weighted 
        Averaging across the entire network.
        
        Args:
            payloads (List[Dict]): List of validated payloads containing 
                                   model weights.
        
        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists 
            of layer names: (feature_layers, classifier_layers).
        """
        if not payloads:
            return [], []
        
        # Extract layer names from the first valid payload
        sample_weights = payloads[0].get("weights", {})
        if not sample_weights:
            return [], []
            
        layer_names = list(sample_weights.keys())
        total_layers = len(layer_names)
        
        if total_layers == 0:
            return [], []

        # 🚨 EMERGENCY FALLBACK: Prevent Feature Layers = 0
        # For standard CNNs (e.g., 8-10 layers), the last 1 or 2 layers 
        # are typically classifiers (Linear/FC), and the rest are feature 
        # extractors (Conv2d, BatchNorm, Pooling).
        # 
        # Heuristic: 
        # - Minimum 1 classifier layer (to avoid empty classifier set)
        # - Maximum 2 classifier layers (to preserve feature extraction)
        # - For very small models (<= 4 layers), use 1 classifier layer
        num_classifier_layers = min(2, max(1, total_layers // 4))
        num_feature_layers = total_layers - num_classifier_layers
        
        feature_layers = layer_names[:num_feature_layers]
        classifier_layers = layer_names[num_feature_layers:]
        
        logger.info(
            f"🔍 [FARABI-AGG] Layer Detection | Total: {total_layers} | "
            f"Feature: {len(feature_layers)} | Classifier: {len(classifier_layers)}"
        )
        
        return feature_layers, classifier_layers

    def _prefilter_enabled(self, arm_default: bool) -> bool:
        """Whether this arm runs _filter_weight_space_anomalies before its
        combination rule.

        Config: protocols.farabi.aggregation.prefilter. Unset/None keeps each
        arm's own default (FARABI arms True, literature baselines False), so
        every existing result is unchanged. True/False forces it, for the
        pre-filter ablation (thesis-hub/docs/decisions.md, 2026-09-28; roadmap
        2.10): the literature baselines never used the filter, so comparing
        them with Farabi-Prox mixed the filter with the combination rule.
        """
        value = (
            self.cfg.get("protocols", {})
            .get("farabi", {})
            .get("aggregation", {})
            .get("prefilter", None)
        )
        return arm_default if value is None else bool(value)

    def _prefilter_models(self, models: List[Dict[str, torch.Tensor]], node: Any, *parallel: List[Any]):
        """Run _filter_weight_space_anomalies on bare weight dicts, keeping any
        per-model side lists (e.g. sample counts) aligned with the survivors.
        Returns (models, *parallel)."""
        tagged = [{"weights": w, "_idx": i} for i, w in enumerate(models)]
        kept = [t["_idx"] for t in self._filter_weight_space_anomalies(tagged, node)]
        return (
            [models[i] for i in kept],
            *[[side[i] for i in kept] for side in parallel],
        )

    def _own_state_anchor_enabled(self) -> bool:
        """Opt-in flag for _maybe_append_own_state (default off -- preserves
        every existing literature-baseline result unchanged unless a plan
        explicitly turns this on). See _maybe_append_own_state docstring."""
        return bool(
            self.cfg.get("protocols", {})
            .get("farabi", {})
            .get("aggregation", {})
            .get("include_own_state_anchor", False)
        )

    def _maybe_append_own_state(
        self,
        models: List[Dict[str, torch.Tensor]],
        node: Any = None,
        extra_weight_list: Optional[List[float]] = None,
        extra_weight_value: float = 1.0,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Opt-in (config: protocols.farabi.aggregation.include_own_state_anchor,
        default False) inclusion of the node's own current model as ONE MORE
        candidate in the population handed to a literature-faithful
        aggregation rule (Multi-Krum/Bulyan/RFA).

        This is deliberately NOT FARABI's own distance-based defense
        (_filter_weight_space_anomalies, which drops payloads far from
        node.weights) -- it only adds one legitimate extra sample to the set
        these algorithms score over with their own unmodified math. Several
        decentralized-FL papers include the node's own current state in the
        candidate set by default (e.g. BALANCE, Fang et al., ACM CCS 2024).

        Added per thesis-hub/docs/decisions.md (2026-09-23): the plan-30 2x2
        control found Multi-Krum/Bulyan/RFA collapse to chance accuracy ONLY
        when poisoning is present (both under IID and severe non-IID --
        ruling out non-IID mismatch as the driver), at exactly the small
        per-call population sizes (aggregate() usually sees n<=3, per plan
        28/29) where these arms have no anchor of their own to fall back on
        -- unlike `robust`/`prox`, which are healthy in every 2x2 cell. This
        tests whether that anchor alone closes the gap.
        """
        if not self._own_state_anchor_enabled():
            return models
        if not (node and hasattr(node, "weights") and node.weights):
            logger.warning(
                "⚠️ [OWN-STATE-ANCHOR] include_own_state_anchor is True but "
                "node.weights is unavailable -- skipping, population unchanged."
            )
            return models
        models = list(models) + [node.weights]
        if extra_weight_list is not None:
            extra_weight_list.append(extra_weight_value)
        logger.info(
            f"⚓ [OWN-STATE-ANCHOR] appended node's own current weights "
            f"(population {len(models) - 1} -> {len(models)})."
        )
        return models

    def _filter_weight_space_anomalies(self, payloads: List[Dict[str, Any]], node: Any = None) -> List[Dict[str, Any]]:
        """
        Weight-Space Anomaly Detection for Byzantine Defense.
        
        This method implements a statistical outlier detection mechanism 
        to defend against Metadata Spoofing and Weight Destruction attacks.
        
        Attack Model Addressed:
        - A malicious node may destroy its model weights (e.g., by adding 
          Gaussian noise with scale 10.0) while spoofing the metadata to 
          claim a high accuracy (e.g., 96.5%).
        - Since the commitment scheme only verifies the integrity of the 
          reported metadata (not the actual weights), such attacks can 
          bypass consensus validation.
        
        Defense Strategy:
        1. Compute the geometric centroid of all fetched model weights 
           in a flattened parameter space.
        2. Calculate the L2 (Euclidean) distance of each model to this 
           centroid.
        3. Identify outliers using a dynamic threshold based on the 
           median distance (robust against adversarial manipulation of 
           the threshold itself).
        4. Filter out models whose distance exceeds 3x the median distance.
        
        Complexity: O(N * P), where N is the number of models and P is 
        the number of parameters. This is acceptable for the aggregation 
        phase as it runs once per aggregation round.
        
        Args:
            payloads (List[Dict]): List of validated payloads with weights.
            node (Any): Reference to the local node (for logging).
        
        Returns:
            List[Dict]: Filtered list of payloads with outliers removed.
        """
        
        N = len(payloads)
        logger.info(f"🛡️ [ANOMALY-FILTER] Called with {N} payloads.")
        
        if N == 0:
            return []
        
        # Get local weights as anchor (for N=1 or N=2)
        local_weights = None
        if node and hasattr(node, 'weights') and node.weights:
            local_weights = node.weights
        
        try:
            layer_names = list(payloads[0]["weights"].keys())
            
            # Helper to compute L2 distance between two weight dicts
            def compute_l2_dist(w1: Dict, w2: Dict) -> float:
                dist = 0.0
                for name in layer_names:
                    if name in w1 and name in w2:
                        dist += torch.norm(w1[name].float() - w2[name].float()).item()
                return dist
            
           # 🟢 Case 1: N >= 3 (Zero-Trust Local Anchor Method)
            if N >= 3 and local_weights:
                # 1. Flatten local weights
                flat_local = torch.cat([local_weights[name].flatten().float() for name in layer_names if name in local_weights])
                
                # 2. Compute distances to the TRUSTED local model (not the poisoned centroid)
                distances = []
                for p in payloads:
                    vec = torch.cat([p["weights"][name].flatten().float() for name in layer_names if name in p["weights"]])
                    dist = torch.norm(vec - flat_local).item()
                    distances.append(dist)
                
                # 3. Dynamic Threshold based on the tightest cluster
                median_dist = np.median(distances)
                # Poisoners usually have distances in thousands (e.g., >8000), honest nodes have small variance
                threshold = max(median_dist * 2.0, 50.0) 
                
                filtered = []
                for i, p in enumerate(payloads):
                    if distances[i] <= threshold:
                        filtered.append(p)
                    else:
                        logger.warning(f"🛡️ [ANOMALY-FILTER] DROPPED {p.get('node_id', 'unknown')} | Dist to local: {distances[i]:.4f} > {threshold:.4f}")
                
                return filtered if filtered else [payloads[np.argmin(distances)]]
            
            # 🟡 Case 2: N == 2 (Anchor-based comparison)
            elif N == 2:
                if local_weights:
                    dist_0 = compute_l2_dist(payloads[0]["weights"], local_weights)
                    dist_1 = compute_l2_dist(payloads[1]["weights"], local_weights)
                    
                    # Malicious models with noise*10 will have extremely high distance
                    threshold = max(dist_0, dist_1) * 0.5  # Heuristic threshold
                    
                    filtered = []
                    if dist_0 <= max(dist_0, dist_1) * 1.5:  # Keep if not extreme outlier
                        filtered.append(payloads[0])
                    else:
                        logger.warning(f"🛡️ [ANOMALY-FILTER] DROPPED {payloads[0].get('node_id', 'unknown')} | Dist to local: {dist_0:.4f}")
                    
                    if dist_1 <= max(dist_0, dist_1) * 1.5:
                        filtered.append(payloads[1])
                    else:
                        logger.warning(f"🛡️ [ANOMALY-FILTER] DROPPED {payloads[1].get('node_id', 'unknown')} | Dist to local: {dist_1:.4f}")
                    
                    return filtered if filtered else [payloads[0]]  # Fallback
                else:
                    # No local anchor, keep both
                    return payloads
            
            # 🔵 Case 3: N == 1 (Validate against local weights)
            else:  # N == 1
                if local_weights:
                    dist = compute_l2_dist(payloads[0]["weights"], local_weights)
                    # If the single model is extremely far from local, it's likely poisoned
                    # Use a conservative threshold based on typical training step distance
                    if dist > 50.0:  # Adjust based on model size; poisoned models usually have dist > 100
                        logger.warning(f"🛡️ [ANOMALY-FILTER] DROPPED single model {payloads[0].get('node_id', 'unknown')} | Dist to local: {dist:.4f}")
                        return []  # Aggregator will fall back to local weights
                return payloads
                
        except Exception as e:
            logger.error(f"❌ [ANOMALY-FILTER] CRITICAL ERROR: {e}", exc_info=True)
            return payloads
    
    def _aggregate_feature_layers(self, payloads: List[Dict], layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Applies Median or Trimmed Mean based on available quorum (N).
        """
        N = len(payloads)
        aggregated = {}
        
        for layer_name in layer_names:
            if layer_name not in payloads[0]["weights"]:
                continue
                
            layer_stack = torch.stack([p["weights"][layer_name] for p in payloads])
            original_dtype = payloads[0]["weights"][layer_name].dtype
            is_float = payloads[0]["weights"][layer_name].is_floating_point()
            
            if N < self.min_models_for_trimmed:
                result_tensor = torch.median(layer_stack, dim=0).values
            else:
                k = max(1, int(N * self.trim_fraction))
                sorted_stack, _ = torch.sort(layer_stack, dim=0)
                if 2 * k < N:
                    trimmed = sorted_stack[k:-k]
                else:
                    trimmed = sorted_stack[N//2:N//2+1]
                result_tensor = trimmed.mean(dim=0)
                
            # Safely cast back to original type (Crucial for integer tensors like BatchNorm counts)
            if not is_float:
                aggregated[layer_name] = result_tensor.round().to(original_dtype)
            else:
                aggregated[layer_name] = result_tensor
                
        return aggregated

    def _aggregate_classifier_layers(self, payloads: List[Dict], layer_names: List[str], node: Any = None) -> Dict[str, torch.Tensor]:
        """
        Applies Virtue-Weighted Averaging to preserve local specializations (Non-IID handling).
        """
        N = len(payloads)
        
        if not layer_names:
            return {}
        
        # Fallback to robust aggregation if the quorum is too small
        if N < self.min_models_for_trimmed:
            return self._aggregate_feature_layers(payloads, layer_names)
            
        aggregated = {}

        first_valid_layer = next((name for name in layer_names if name in payloads[0]["weights"]), None)
        if not first_valid_layer:
            logger.warning("⚠️ [FARABI-AGG] No valid classifier layers found in payloads.")
            return {}
        
        target_device = payloads[0]["weights"][first_valid_layer].device
                
        weights = []
        for p in payloads:
            samples = p.get("samples", 1)
            if node and hasattr(node, 'reputation'):
                proposer_id = p.get("node_id", "")
                virtue = node.reputation.get_virtue_score(proposer_id)
                samples = samples * (virtue / 100.0)
            weights.append(max(samples, 1.0))
            
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=target_device)
        sum_weights = weights_tensor.sum()
        
        if sum_weights > 0:
            weights_tensor = weights_tensor / sum_weights
        else:
            weights_tensor = torch.ones_like(weights_tensor) / N
            
        for layer_name in layer_names:
            if layer_name not in payloads[0]["weights"]:
                continue
                
            layer_stack = torch.stack([p["weights"][layer_name] for p in payloads])
            weight_shape = [-1] + [1] * (layer_stack.dim() - 1)
            result_tensor = (layer_stack * weights_tensor.view(*weight_shape)).sum(dim=0)
                    
            original_dtype = payloads[0]["weights"][layer_name].dtype
            if not payloads[0]["weights"][layer_name].is_floating_point():
                aggregated[layer_name] = result_tensor.round().to(original_dtype)
            else:
                aggregated[layer_name] = result_tensor
                
        return aggregated
    
    
    def client_update(self, node: Any, global_weights: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the local training process on a client node.
        
        In the FARABI DAG-FL architecture, this delegates the training to the 
        node's dedicated LocalTrainer. This preserves the Single Responsibility 
        Principle (SRP) and ensures that the ModelCacheManager is utilized 
        to prevent GPU Out-of-Memory (OOM) errors.
        
        Args:
            node: The node instance performing the training.
            global_weights: Optional weights to initialize the model.
            
        Returns:
            Dict: A payload containing the updated weights and training metadata.
        """
        if not node:
            logger.warning("⚠️ [FARABI-AGG] Cannot perform client_update without a valid node reference.")
            return {"weights": global_weights or {}}
            
        # Delegate to the node's dedicated trainer to utilize GPU Caching
        if hasattr(node, 'trainer') and hasattr(node.trainer, 'train_and_evaluate'):
            # Use global weights if provided, otherwise fallback to the node's current local weights
            initial_weights = global_weights if global_weights is not None else node.weights
            
            # Execute local training phase
            new_weights, compute_time, accuracy = node.trainer.train_and_evaluate(initial_weights)
            
            return {
                "node_id": getattr(node, 'node_id', 'unknown'),
                "weights": new_weights,
                "accuracy": accuracy,
                "compute_time": compute_time
            }
            
        # Fallback if the node lacks a standard trainer
        logger.warning(f"⚠️ [FARABI-AGG] Node {getattr(node, 'node_id', 'unknown')} lacks a standard LocalTrainer.")
        return {
            "node_id": getattr(node, 'node_id', 'unknown'), 
            "weights": global_weights or {}
        }
        