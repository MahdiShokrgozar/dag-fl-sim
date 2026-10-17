"""
Node Factory Module for Federated Learning Simulator.

This module is responsible for instantiating simulation nodes. It handles node 
type distribution, data sharding, and dynamically injects hardware heterogeneity.
"""

import copy
import logging
import random
import numpy as np  # 💡 Added for Pareto distribution
import torch
from typing import Dict, Any, List, Type, Optional, Tuple

# Base Node Import
from src.nodes.base_node import BaseNode

# Malicious Nodes Imports
from src.nodes.malicious.stealing import StealingNode
from src.nodes.malicious.lazy import LazyNode
from src.nodes.malicious.poisoning import PoisoningNode
from src.nodes.malicious.label_flipping import LabelFlippingNode
from src.nodes.malicious.backdoor import BackdoorNode

logger = logging.getLogger("NodeFactory")

class SimpleDataset:
    """
    A lightweight wrapper to make data compatible with FastTensorDataLoader.
    Ensures .data and .targets attributes exist.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class NodeFactory:
    """
    Factory class responsible for instantiating simulation nodes.
    Handles node type distribution, dataset extraction, and hardware profiling.
    """

    NODE_TYPES: Dict[str, Type[BaseNode]] = {
        "honest": BaseNode,
        "stealing": StealingNode,
        "lazy": LazyNode,
        "poisoning": PoisoningNode,
        "flipping": LabelFlippingNode,
        "backdoor": BackdoorNode
    }

    @staticmethod
    def _generate_hardware_profile(config: Dict[str, Any]) -> Tuple[float, bool, float]:
        """
        Generates heterogeneous hardware capacities and straggler penalties.
        Supports multiple statistical distributions (pareto, normal, uniform, constant) 
        driven dynamically by the simulation configuration.

        Args:
            config (Dict[str, Any]): Global simulation configuration.

        Returns:
            Tuple[float, bool, float]: (capacity, is_straggler, delay_multiplier)
        """
        # --- 1. Hardware Capacity Injection ---
        hw_cfg = config.get("network", {}).get("hardware_heterogeneity", {})
        capacities = [0.5, 1.0, 4.0]
        weights = [
            float(hw_cfg.get("weak_nodes_ratio", 0.0)),
            float(hw_cfg.get("standard_nodes_ratio", 1.0)), 
            float(hw_cfg.get("strong_nodes_ratio", 0.0))
        ]
        
        if sum(weights) == 0:
            weights = [0.0, 1.0, 0.0]
            
        assigned_capacity = random.choices(capacities, weights=weights, k=1)[0]

        # --- 2. Dynamic Straggler Injection ---
        straggler_cfg = config.get("straggler", {})
        is_enabled = straggler_cfg.get("enabled", False)
        
        is_straggler = False
        delay_multiplier = 1.0

        if is_enabled:
            dist_type = straggler_cfg.get("distribution", "pareto").lower()
            params = straggler_cfg.get("params", {})
            
            # Dynamically apply the requested statistical distribution
            if dist_type == "pareto":
                alpha = float(params.get("alpha", 3.0))
                # np.random.pareto is [0, inf). Adding 1.0 ensures minimum 1x baseline.
                delay_multiplier = np.random.pareto(alpha) + 1.0
                
            elif dist_type == "normal":
                mu = float(params.get("mu", 2.0))
                sigma = float(params.get("sigma", 0.5))
                # Ensure multiplier never drops below 1.0 (cannot be faster than its own capacity)
                delay_multiplier = max(1.0, np.random.normal(mu, sigma))
                
            elif dist_type == "uniform":
                low = float(params.get("low", 1.0))
                high = float(params.get("high", 5.0))
                delay_multiplier = np.random.uniform(low, high)
                
            elif dist_type == "constant":
                delay_multiplier = float(params.get("value", 3.0))
                
            else:
                logger.warning(f"Unknown straggler distribution '{dist_type}'. Defaulting to multiplier 1.0")
                delay_multiplier = 1.0

            # 💡 Dynamic Threshold: Define what multiplier officially qualifies a node as a straggler
            threshold = float(straggler_cfg.get("straggler_threshold", 2.0))
            if delay_multiplier >= threshold:
                is_straggler = True

        return assigned_capacity, is_straggler, delay_multiplier

    @staticmethod
    def create_nodes(
        config: Dict[str, Any],
        train_dataset: Any,
        test_dataset: Any,
        protocol: Any,
        cache_manager: Any,
        storage_engine: Any,
        model_template: Any = None,
        simulator_ref: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, BaseNode]:
        
        num_nodes = config['network']['num_nodes']
        device = config['simulation']['device']
        
        # 1. Determine Node Types
        assigned_types = NodeFactory._assign_node_types(config, num_nodes)
        
        nodes: Dict[str, BaseNode] = {}
        
        # 2. Prepare Weights (Cold Start)
        if model_template is not None:
            initial_weights = copy.deepcopy(model_template.state_dict())
        else:
            logger.warning("No model_template provided. Creating temporary model.")
            from src.models.factory import ModelFactory
            temp_model = ModelFactory.create(config)
            initial_weights = temp_model.state_dict()
            del temp_model

        # 3. Robust Dataset Extraction (Maintained your exact logic)
        all_data, all_targets = None, None
        is_pre_sharded = False
        dataset_type = type(train_dataset).__name__
        
        if isinstance(train_dataset, dict):
            data_keys = ['data', 'tensors', 'x', 'features', 'images']
            target_keys = ['targets', 'labels', 'y', 'class']
            found_data = next((k for k in data_keys if k in train_dataset), None)
            found_target = next((k for k in target_keys if k in train_dataset), None)
            
            if found_data and found_target:
                all_data = train_dataset[found_data]
                all_targets = train_dataset[found_target]
            else:
                first_key = next(iter(train_dataset))
                if isinstance(first_key, int) or (isinstance(first_key, str) and "node" in first_key):
                    is_pre_sharded = True
                    logger.info("Detected pre-sharded dataset dictionary.")
                    
        elif hasattr(train_dataset, 'tensors'):
            all_data, all_targets = train_dataset.tensors
        elif isinstance(train_dataset, (tuple, list)) and len(train_dataset) == 2:
            all_data, all_targets = train_dataset
        elif hasattr(train_dataset, 'data') and hasattr(train_dataset, 'targets'):
            all_data = train_dataset.data
            all_targets = train_dataset.targets

        if all_data is None and not is_pre_sharded:
             raise ValueError(f"Dataset format '{dataset_type}' not recognized.")

        if not is_pre_sharded:
            total_samples = len(all_data)
            samples_per_node = total_samples // num_nodes
            logger.info(f"Data Sharding (IID): ~{samples_per_node} samples per node (Total: {total_samples})")

        # 4. Instantiate Nodes
        for i in range(num_nodes):
            node_id = f"node_{i}"
            node_type_str = assigned_types[i]
            
            node_class = NodeFactory.NODE_TYPES.get(node_type_str, BaseNode)
            if node_type_str not in NodeFactory.NODE_TYPES:
                logger.warning(f"Unknown type '{node_type_str}'. Defaulting to Honest.")

            is_malicious = node_type_str != "honest"

            # --- 💡 NEW: Generate Hardware & Straggler profile for this node ---
            node_capacity, is_straggler, delay_multiplier = NodeFactory._generate_hardware_profile(config)

            if node_capacity >= 4.0:
                hardware_tier = "Edge Server"
            elif node_capacity >= 1.0:
                hardware_tier = "Standard Node"
            else:
                hardware_tier = "Mobile/IoT"
                
            # 5. Assign Data Shard (Maintained your exact logic)
            node_dataset = None
            if is_pre_sharded:
                if i in train_dataset:
                    raw_shard = train_dataset[i]
                elif node_id in train_dataset:
                    raw_shard = train_dataset[node_id]
                else:
                    raise ValueError(f"Shard for {node_id} not found.")
                
                if isinstance(raw_shard, (tuple, list)):
                    node_dataset = SimpleDataset(raw_shard[0], raw_shard[1])
                elif hasattr(raw_shard, 'data') and hasattr(raw_shard, 'targets'):
                    node_dataset = raw_shard
                elif hasattr(raw_shard, 'tensors'):
                    node_dataset = SimpleDataset(raw_shard.tensors[0], raw_shard.tensors[1])
                elif isinstance(raw_shard, dict) and 'x' in raw_shard:
                     node_dataset = SimpleDataset(raw_shard['x'], raw_shard['y'])
                else:
                     node_dataset = raw_shard
            else:
                start_idx = i * samples_per_node
                end_idx = start_idx + samples_per_node if i < num_nodes - 1 else len(all_data)
                node_dataset = SimpleDataset(all_data[start_idx:end_idx], all_targets[start_idx:end_idx])

            # test_dataset is either a single shared Dataset (CentralizedSimulator
            # - a real server legitimately has one global view) or a dict of
            # per-node local test partitions keyed like train_dataset
            # (DecentralizedSimulator/L2ScalingSimulator, since 2026-09-15 -
            # see thesis-hub/docs/decisions.md, same date). Resolve here so
            # every node_class constructor call stays unchanged either way.
            if isinstance(test_dataset, dict):
                node_test_set = test_dataset.get(node_id, test_dataset.get(i))
                if node_test_set is None:
                    raise ValueError(f"Local test partition for {node_id} not found.")
            else:
                node_test_set = test_dataset

            try:
                node = node_class(
                    node_id=node_id,
                    initial_weights=copy.deepcopy(initial_weights),
                    data_subset=node_dataset,
                    test_set=node_test_set,
                    config=config,
                    aggregator=protocol.get_aggregator(),
                    tip_selector=protocol.get_tip_selector(),
                    consensus=protocol.get_consensus_strategy(),
                    batch_policy=protocol.get_batch_policy(),
                    processor=protocol.get_update_processor(),
                    cache_manager=cache_manager,
                    l2_strategy=protocol.get_l2_strategy(),
                    reputation=protocol.get_reputation_module(),
                    capacity=node_capacity, # 💡 INJECTED CAPACITY
                    device=device,
                    capability_tag="Normal",
                    is_malicious=is_malicious,
                    storage_engine=storage_engine,
                    simulator_ref=simulator_ref                    
                )
                
                # 💡 INJECT STRAGGLER ATTRIBUTES
                node.node_type = node_type_str
                node.is_malicious = is_malicious
                node.is_straggler = is_straggler
                node.delay_multiplier = delay_multiplier
                node.hardware_tier = hardware_tier    
                node.is_straggler = is_straggler    
                node.delay_multiplier = delay_multiplier
                
                if node.consensus and hasattr(node.consensus, 'initialize'):
                    node.consensus.initialize(node)
                
                nodes[node_id] = node
                
                # Optional: Log severe stragglers for debugging
                if is_straggler:
                    logger.debug(f"[{node_id}] instantiated as a Straggler! Delay Multiplier: {delay_multiplier:.2f}x")
                
            except Exception as e:
                logger.error(f"Critical error instantiating {node_id} ({node_type_str}): {str(e)}")
                raise ValueError(f"Failed to create node {node_id}") from e

        logger.info(f"Successfully created {len(nodes)} nodes with hardware profiling applied.")
        return nodes

    @staticmethod
    def _assign_node_types(config: Dict[str, Any], num_nodes: int) -> List[str]:
        # (Maintained your exact logic)
        participants_cfg = config.get("participants", {})
        distribution = participants_cfg.get("distribution", {"honest": 1.0})
        
        types_list = []
        allocated_nodes = 0
        total_weight = sum(distribution.values())
        if total_weight == 0:
            distribution = {"honest": 1.0}
            total_weight = 1.0
            
        for node_type, weight in distribution.items():
            count = int(num_nodes * (weight / total_weight))
            types_list.extend([node_type] * count)
            allocated_nodes += count
            
        remainder = num_nodes - allocated_nodes
        if remainder > 0:
            largest_group = max(distribution, key=distribution.get)
            types_list.extend([largest_group] * remainder)
            
        random.shuffle(types_list)
        return types_list