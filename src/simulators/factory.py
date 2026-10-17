import logging

class SimulatorFactory:
    @staticmethod
    def get_simulator(config: dict, metric_manager=None):
        """
        Creates the simulator engine and injects the metric manager..
        """
        logger = logging.getLogger("SimulatorFactory")
        backend = config.get("simulation", {}).get("backend", "decentralized").lower()
        
        sim = None
        
        # 1. Instantiate the Engine (Lazy Loading)
        if backend == "centralized":
            from src.simulators.engines.centralized import CentralizedSimulator
            sim = CentralizedSimulator(config)
            
        elif backend == "decentralized":
            from src.simulators.engines.decentralized import DecentralizedSimulator
            sim = DecentralizedSimulator(config)
            
        elif backend == "l2_scaling":
            from src.simulators.engines.l2_scaling import L2ScalingSimulator
            sim = L2ScalingSimulator(config)
            
        else:
            raise ValueError(f"Unknown simulation backend: {backend}")

        # 2. Inject Metric Manager
        if metric_manager:
            sim.metrics_manager = metric_manager
            metric_manager.attach_source(sim)
            logger.info(f"✅ MetricManager injected into {type(sim).__name__}")
            
        return sim

    @staticmethod
    def create(config: dict):
        return SimulatorFactory.get_simulator(config, None)