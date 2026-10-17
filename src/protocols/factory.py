# src/protocols/factory.py

import logging
from typing import Dict, Any

# --- 1. Explicit Imports of All Strategies ---
# This serves as the "Table of Contents" for your framework.

# Aggregation
from src.strategies.aggregation.fedavg import FedAvg
from src.strategies.aggregation.fedprox import FedProx
from src.strategies.aggregation.krum import Krum
from src.strategies.aggregation.median import Median
from src.strategies.aggregation.scaffold import ScaffoldAggregator
from src.strategies.aggregation.trimmed_mean import TrimmedMean
from src.strategies.aggregation.weighted_fedavg import WeightedFedAvg

# Consensus
from src.strategies.consensus.simple import SimpleConsensus
from src.strategies.consensus.baseline_por_consensus import BaselinePoRConsensus


# Selection
from src.strategies.selection.random_selection import RandomTipSelector
from src.strategies.selection.mcmc_selection import MCMCSelection
from src.strategies.selection.latest_selection import LatestTipSelector

# Batching
from src.strategies.batching.no_batch import NoBatchPolicy
from src.strategies.batching.size_based import CountBasedBatching

# Processing
from src.strategies.processing.identity import IdentityProcessor
from src.strategies.processing.compression import GzipProcessor

# Layer 2
from src.strategies.l2.clustering import RandomClustering

# Protocols
from src.protocols.generic import GenericProtocol
from src.protocols.ironforge import IronForgeProtocol
from src.protocols.dbfl import DBFLProtocol
from src.protocols.farabi import FarabiProtocol


class ProtocolFactory:
    """
    Central Factory that maps string names (from YAML) to Python Classes.
    """

    # --- 2. Registry Maps ---
    
    AGGREGATORS = {
        "fedavg": FedAvg,
        "weighted_fedavg": WeightedFedAvg,
        "fedprox": FedProx,
        "scaffold": ScaffoldAggregator,
        "median": Median,
        "trimmed_mean": TrimmedMean,
        "krum": Krum
    }

    CONSENSUS = {
        "simple": SimpleConsensus,
        "none": SimpleConsensus,
        "baseline_por": BaselinePoRConsensus
    }

    TIP_SELECTORS = {
        "random": RandomTipSelector,
        "mcmc": MCMCSelection,
        "latest": LatestTipSelector,
    }

    BATCHING = {
        "none": NoBatchPolicy,
        "size_based": CountBasedBatching
    }

    PROCESSING = {
        "identity": IdentityProcessor,
        "compression": GzipProcessor
    }

    L2_STRATEGIES = {
        "random_clustering": RandomClustering
    }

    PROTOCOLS = {
        "generic_protocol": GenericProtocol,
        "ironforge": IronForgeProtocol,
        "dbfl": DBFLProtocol,
        "farabi": FarabiProtocol
    }

    @staticmethod
    def create(config: Dict[str, Any]):
        logger = logging.getLogger("ProtocolFactory")

        # 1. Determine which Protocol Flow to use
        proto_name = config.get("simulation", {}).get("protocol", "generic_protocol")
        ProtocolClass = ProtocolFactory.PROTOCOLS.get(proto_name)

        if not ProtocolClass:
            raise ValueError(f"❌ Protocol '{proto_name}' not found in ProtocolFactory.")

        # If it's a specific paper implementation (like IronForge), it might handle its own strategies.
        # But if it's GenericProtocol, we inject everything.
        
        if proto_name != "generic_protocol":
            # Specific protocols usually initialize their own defaults or read config internally
            return ProtocolClass(config)

        # 2. For GenericProtocol, we assemble the components:
        
        def load_strat(map_dict, section, key, default="none"):
            name = config.get(section, {}).get(key, default)
            cls = map_dict.get(name)
            if cls:
                return cls(config) # Instantiate with config
            if name not in ["none", ""]:
                logger.warning(f"⚠️ Strategy '{name}' not found for {key}. Using None.")
            return None

        # Instantiate components
        # Note: Adjust YAML keys to match your config structure (e.g., 'protocol' section)
        aggregator = load_strat(ProtocolFactory.AGGREGATORS, "protocol", "aggregator", "fedavg")
        consensus = load_strat(ProtocolFactory.CONSENSUS, "protocol", "consensus", "none")
        tip_selector = load_strat(ProtocolFactory.TIP_SELECTORS, "protocol", "tip_selection", "random")
        batching = load_strat(ProtocolFactory.BATCHING, "protocol", "batching", "none")
        processor = load_strat(ProtocolFactory.PROCESSING, "protocol", "processing", "identity")
        
        # L2 is usually in its own section
        l2_strategy = None
        if config.get("layer2", {}).get("enabled"):
            l2_name = config.get("layer2", {}).get("strategy")
            L2Class = ProtocolFactory.L2_STRATEGIES.get(l2_name)
            if L2Class:
                l2_strategy = L2Class(config.get("layer2", {}).get("params", {}))

        logger.info(f"✅ GenericProtocol Assembled: Agg={type(aggregator).__name__}, Cons={type(consensus).__name__}, L2={type(l2_strategy).__name__}")

        return GenericProtocol(
            config=config,
            aggregator=aggregator,
            consensus=consensus,
            tip_selector=tip_selector,
            batch_policy=batching,
            update_processor=processor,
            l2_strategy=l2_strategy
        )