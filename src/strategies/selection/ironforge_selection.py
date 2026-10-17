"""
IronForge Tip Selection Strategy Module.

This module implements the candidate selection process as described in 
Algorithm 2 of the IRONFORGE paper. It evaluates a subset of available DAG 
tips to select the optimal parent models for aggregation.

To ensure scalable simulation, this implementation uses an analytical time 
modeling approach combined with a global accuracy oracle. This bypasses the 
computational bottleneck of explicit GPU-based model evaluations during 
the simulation's virtual timeline, while maintaining rigorous scientific validity.
"""

import random
from typing import List, Dict, Any, Tuple
from src.strategies.selection.base import TipSelector


class IronForgeTipSelector(TipSelector):
    """
    Tip selection strategy for the IronForge protocol.
    
    Attributes:
        beta (int): The number of candidate models to sample and evaluate.
        sigma (int): The final number of best parent models to select.
        base_inference_time (float): Base time (in seconds) required to process one sample.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the selector dynamically using the global configuration.

        Args:
            config (Dict[str, Any]): The global simulation configuration dictionary.
        """
        super().__init__(config)
        
        # Extract IronForge specific selection parameters
        protocols_cfg = config.get("protocols", {})
        if_cfg = protocols_cfg.get("ironforge", config.get("ironforge", {}))
        selection_cfg = if_cfg.get("selection", {})
        
        self.beta = selection_cfg.get("beta", 10)
        self.sigma = selection_cfg.get("sigma", 2)
        
        # Base parameter for analytical delay modeling
        self.base_inference_time = self.config.get("local_training", {}).get("base_inference_time_per_sample", 0.0001)

    def select_tips(self, dag: Any, node: Any) -> Tuple[List[str], float]:
        """
        Selects parent nodes based on IronForge logic and calculates simulated delay.
        
        According to Algorithm 2 of IRONFORGE, workers randomly select \beta nodes 
        and evaluate them using their local testing dataset.
        
        Args:
            dag (Any): The global or local DAG instance containing transactions.
            node (BaseNode): The worker node executing the tip selection.

        Returns:
            Tuple[List[str], float]: 
                - A list of selected transaction hashes (parents).
                - The simulated analytical delay (in seconds) required for evaluation.
        """
        raw_available_tips = list(dag.tips)
        simulator_oracle = getattr(node.simulator, "oracle_state", {})

        # Prevent nodes that were economically slashed by the L2 Settlement from being selected.
        available_tips = []
        for tx_hash in raw_available_tips:
            true_state = simulator_oracle.get(tx_hash, {})
            author_id = true_state.get("author", None)
            
            is_slashed = False
            if author_id and hasattr(node.simulator, "nodes"):
                author_node = node.simulator.nodes.get(author_id)
                # Check the flag we set in ironforge_settlement.py
                if author_node and getattr(author_node, "is_slashed", False):
                    is_slashed = True
                    
            if not is_slashed:
                available_tips.append(tx_hash)

        # 0. Handle genesis or empty DAG scenario
        if not available_tips:
            return ([dag.genesis['hash']] if hasattr(dag, 'genesis') and dag.genesis else []), 0.0

        # 1. Sample \beta candidates securely
        sample_size = min(self.beta, len(available_tips))
        candidates_hashes = random.sample(available_tips, sample_size)

        # 2. Evaluate candidates using the Simulator's Rich Oracle
        evaluated_candidates = []
        
        for tx_hash in candidates_hashes:
            # Fetch ground-truth state to prevent evaluation manipulation by malicious nodes
            true_state = simulator_oracle.get(tx_hash, {})
            true_accuracy = true_state.get("accuracy", 0.0)
            
            # Simulate local dataset variance by adding controlled uniform noise (e.g., ±2
            # percentage points - true_accuracy is on the 0-100 scale, see
            # LocalTrainer.evaluate_state).
            local_variance_noise = random.uniform(-2.0, 2.0)
            # Clamp to the SAME 0-100 scale as true_accuracy. This used to clamp
            # to [0.0, 1.0], a fraction-scale bound applied to a percentage-scale
            # value: every candidate with true_accuracy above ~1% (i.e. anything
            # better than a poisoning node's destroyed model) saturated to
            # exactly 1.0, so "rank descending by estimated accuracy" (the whole
            # point of Algorithm 2's sigma-selection) was a tie across nearly the
            # entire candidate pool and picked among them arbitrarily instead of
            # by merit. See thesis-hub/docs/decisions.md, 2026-09-15.
            local_accuracy_estimation = max(0.0, min(100.0, true_accuracy + local_variance_noise))
            
            evaluated_candidates.append((tx_hash, local_accuracy_estimation))

        # 3. Rank candidates descending by estimated accuracy and select the top \sigma models
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = min(self.sigma, len(evaluated_candidates))
        selected_tips = [tx_hash for tx_hash, _ in evaluated_candidates[:top_k]]

        # 4. Analytical Delay Modeling using Dynamic Capabilities
        node_bandwidth = node.capabilities.get("bandwidth_mbps", 10.0)
        
        # We assume evaluating beta models requires downloading them first
        model_size_mb = self.config.get("model", {}).get("estimated_size_mb", 5.0)
        download_time_per_model = (model_size_mb * 8) / node_bandwidth
        
        base_capacity = getattr(node, 'capacity', 1.0)
        delay_multiplier = getattr(node, 'delay_multiplier', 1.0)
        effective_capacity = base_capacity / delay_multiplier
        
        # In IronForge, evaluating candidate parent models should be a lightweight operation.
        eval_batch_size = 64 
        inference_time_per_sample = self.base_inference_time / max(0.01, effective_capacity)
        evaluation_time_per_model = eval_batch_size * inference_time_per_sample
        
        # Total delay calculation: Downloading and evaluating \beta models
        total_download_delay = sample_size * download_time_per_model
        total_eval_delay = sample_size * evaluation_time_per_model
        total_bottleneck_delay = total_download_delay + total_eval_delay

        # Book the two halves to the channels they actually belong to.
        #
        # Both used to go to register_compute(), which broke cross-protocol
        # comparison in two directions at once: IronForge's CPU figure absorbed
        # time that is really network transfer, and -- because no register_traffic
        # call existed anywhere in the IronForge strategies -- IronForge reported
        # ZERO IPFS/model bytes. Any bandwidth-overhead table built on that would
        # have shown the replay-heavy baseline as costing no bandwidth while
        # FARABI, which books its fetches properly in
        # FarabiConsensus._fetch_data_from_ipfs, paid for every byte.
        # Same formula and same channel name as that method, so the two
        # protocols are now measured on one basis.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        if hasattr(node, 'telemetry'):
            node.telemetry.register_compute(total_eval_delay)
            node.telemetry.register_network_delay(total_download_delay)
            downloaded_bytes = int(sample_size * model_size_mb * 1024 * 1024)
            node.telemetry.register_traffic(downloaded_bytes, channel="ipfs")

        # Fallback mechanism if selection fails
        if not selected_tips:
            selected_tips = random.sample(available_tips, min(self.sigma, len(available_tips)))

        return selected_tips, total_bottleneck_delay