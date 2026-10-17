# src/simulators/engines/centralized.py
import random
import copy
import logging
from typing import Dict, Any

import torch

from src.simulators.engines.base_simulator import BaseSimulator, Event
from src.nodes.factory import NodeFactory
from src.data.loader import load_dataset
from src.models import get_model
from src.aggregation import get_aggregator
from src.protocols.factory import ProtocolFactory
from src.services.storage.factory import StorageFactory
from src.utils.gpu_cache import ModelCacheManager
from src.utils.utils import get_object_size_in_bytes
from src.utils.ipfs_utils import download_model_from_ipfs


class CentralizedSimulator(BaseSimulator):
    """
    Standard Centralized Federated Learning Simulator (Client-Server), wired
    into the shared discrete-event loop as a single self-rescheduling
    CENTRALIZED_ROUND event per round - the same pattern L2ScalingSimulator
    uses for its periodic topology/settlement events.

    Rebuilt 2026-09-13 (thesis-hub/docs/decisions.md, same date): this class
    had never actually run against the current node architecture. It hand-
    built BaseNode with a wrong argument list (a live model where a weights
    dict was expected, simulator_ref/aggregator swapped, 5 required args
    missing entirely), referenced self.cfg/self.device which nothing ever
    set, and drove per-client training through node.model/.evaluate()/
    .default_train_model(), none of which exist on BaseNode today. Node
    construction now goes through NodeFactory like every other engine;
    training/eval go through node.trainer (see the matching fix in
    src/strategies/aggregation/fedavg.py's client_update).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger("CentralizedSimulator")
        self.device = torch.device(config.get("simulation", {}).get("device", "cpu"))
        self.nodes: Dict[str, Any] = {}
        self.round_num = 0

    def initialize(self):
        self.logger.info("🏗️ Initializing Centralized Simulation...")

        # 1. Load Data and Model. A real central server genuinely has a
        # global view, unlike a decentralized node - so this engine
        # deliberately keeps using global_test_set for every client, not the
        # per-node local partitions DecentralizedSimulator now uses. See
        # thesis-hub/docs/decisions.md, 2026-09-15.
        shards, _test_shards, test_set = load_dataset(self.config)
        self.test_set = test_set

        self.global_model = get_model(self.config)
        self.global_model.to(self.device)

        self.aggregator = get_aggregator(self.config, device=self.device)

        # 2. Algorithm-specific server state (e.g. SCAFFOLD's control variate)
        self.server_state = {}
        if self.aggregator.agg_type == "scaffold":
            self.logger.info("SCAFFOLD detected. Initializing server control variate.")
            initial_weights = self.global_model.state_dict()
            self.server_state['control_variate'] = {
                k: torch.zeros_like(v, device=self.device) for k, v in initial_weights.items()
            }

        # 3. Create clients the same way every other engine does
        storage = StorageFactory.create(self.config)
        protocol = ProtocolFactory.create(self.config)
        cache_cap = self.config.get("simulation", {}).get("gpu_cache_capacity", 5)
        self.cache_manager = ModelCacheManager(self.global_model, device=self.device, max_capacity=cache_cap)

        self.nodes = NodeFactory.create_nodes(
            config=self.config,
            model_template=self.global_model,
            train_dataset=shards,
            test_dataset=test_set,
            storage_engine=storage,
            protocol=protocol,
            cache_manager=self.cache_manager,
            simulator_ref=self,
        )
        # One node stands in as the server/sequencer for evaluation and for
        # attributing this backend's client<->server traffic - the same role
        # L2ScalingSimulator gives an arbitrary node for its settlement payload.
        self.sequencer_id = next(iter(self.nodes))
        self.logger.info(f"✅ Created {len(self.nodes)} centralized clients.")

        # 4. Schedule the first round.
        #
        # 2026-09-13 (thesis-hub/docs/decisions.md, same date): this used to
        # reuse base_training_time_per_epoch (~1s) - the granularity of a
        # single decentralized node's own training-to-next-round delay. That's
        # the wrong unit here: with participation_ratio 1.0 every node trains
        # every round, so ~1s rounds over a 400s max_time meant ~400 rounds *
        # 12 nodes = ~4800 real local-training calls, plus a full test-set
        # eval every round - about 12x the real compute of a decentralized
        # cell covering the same max_time, and it never finished in a
        # reasonable time on Kaggle.
        #
        # First fix reused reporting.interval (30s -> ~13 rounds), which
        # finished fast but converged to 87.2% - BELOW farabi_honest's 91.9%
        # over the same window, contradicting this cell's purpose as the
        # accuracy ceiling a decentralized protocol is trying to reach. 13
        # synchronous rounds is too few update steps against ~hundreds of
        # staggered async ones. So this is its own dedicated knob, not
        # reporting.interval's job twice over - defaults to reporting.interval
        # only when unset, but a plan can (and here does) ask for finer
        # rounds without changing how often metrics get sampled.
        self.round_interval = self.config.get("simulation", {}).get(
            "centralized_round_interval", self.config.get("reporting", {}).get("interval", 30.0)
        )
        self.schedule_event(timestamp=self.round_interval, event_type="CENTRALIZED_ROUND")

    def _handle_event(self, event: Event):
        if event.event_type == "CENTRALIZED_ROUND":
            self.round_num += 1
            self._run_round(self.round_num)
            next_time = self.current_time + self.round_interval
            if next_time <= self.max_time:
                self.schedule_event(timestamp=next_time, event_type="CENTRALIZED_ROUND")
        else:
            self.logger.warning(f"⚠️ Unknown event type: {event.event_type}")

    def _run_round(self, round_num: int) -> Dict[str, Any]:
        self.logger.debug(f"--- Centralized Round {round_num} (t={self.current_time:.2f}) ---")

        # 1. Client Selection
        ratio = self.config.get("simulation", {}).get("participation_ratio", 1.0)
        num_participants = max(1, int(len(self.nodes) * ratio))
        participants = random.sample(list(self.nodes.values()), num_participants)

        # 2. Downlink (Server -> Clients)
        current_weights = copy.deepcopy(self.global_model.state_dict())
        download_size = get_object_size_in_bytes(current_weights)
        if self.aggregator.agg_type == "scaffold":
            download_size += get_object_size_in_bytes(self.server_state['control_variate'])
        round_bytes = download_size * len(participants)

        # 3. Local Training
        local_payloads = []
        for node in participants:
            if self.aggregator.agg_type == "scaffold":
                payload = self.aggregator.client_update(node, current_weights, self.server_state['control_variate'])
            else:
                payload = self.aggregator.client_update(node, current_weights)

            if payload:
                local_payloads.append(payload)
                round_bytes += get_object_size_in_bytes(payload)

        sequencer = self.nodes[self.sequencer_id]
        if hasattr(sequencer, 'telemetry'):
            sequencer.telemetry.register_traffic(round_bytes, channel="l1")

        # 4. Aggregation
        processed_updates = []
        for p in local_payloads:
            weights = None
            if p.get('model_format') == 'ipfs':
                try:
                    w, _ = download_model_from_ipfs(
                        p['model_cid'], self.config.get("storage", {}), getattr(self, 'mock_ipfs', None)
                    )
                    weights = w
                except Exception:
                    continue
            else:
                weights = p.get('weights')

            if weights:
                p['weights'] = weights
                processed_updates.append(p)

        if processed_updates:
            if self.aggregator.agg_type == "scaffold":
                avg_weights, new_server_c = self.aggregator.aggregate(processed_updates, self.server_state['control_variate'])
                self.global_model.load_state_dict(avg_weights)
                self.server_state['control_variate'] = new_server_c
            else:
                avg_weights = self.aggregator.aggregate(processed_updates)
                self.global_model.load_state_dict(avg_weights)

        # 5. Global Evaluation - evaluate the aggregated model once, then
        # propagate to every node so the shared AccuracyMetric collector
        # (which reads per-node node_accuracy) reports the actual global
        # number rather than each client's stale pre-aggregation accuracy
        # from step 3.
        evaluator = self.nodes[self.sequencer_id]
        eval_model = evaluator.cache_manager.checkout_model(self.global_model.state_dict())
        accuracy, per_class_acc = evaluator.trainer.evaluate_state(eval_model)
        evaluator.cache_manager.checkin_model(eval_model)

        for node in self.nodes.values():
            node.node_accuracy = accuracy
            node.last_accuracy = accuracy
            node.per_class_accuracy = per_class_acc

        self.logger.info(f"Round {round_num} Global Accuracy: {accuracy:.2f}%")

        return {
            "average_accuracy": accuracy,
            "num_participants": len(participants),
            "processed_updates": len(processed_updates),
        }
