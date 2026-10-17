# Developing with DAG-FL-Sim

How a run is put together, and how to add an aggregation rule, an attacker or a validation
variant. Paths are relative to the repository root.

## How a run is assembled

1. **Plan → run configuration.** `run_experiments.py --plan <plan.yaml>` builds one
   configuration per (experiment, arm, seed) by merging, in this order,
   `configs/defaults.yaml`, the plan's top-level keys, the experiment's `base_config` and the
   arm's `overrides` (a recursive dictionary merge: later values win, nested keys are merged
   rather than replaced). The result is written to
   `<output_root>/<experiment>/<arm>/seed_<seed>/run_config.yaml`, and the run is started as a
   separate process, `python main.py --config <that file>`, with `PYTHONHASHSEED` fixed.
   `--jobs N` runs N such processes at once.
2. **Engine.** `main.py` picks the simulator from `simulation.backend`
   (`src/simulators/factory.py`): `l2_scaling` (the DAG ledger used in the paper),
   `decentralized` or `centralized` (a FedAvg reference). All are discrete-event engines:
   training, publishing, network delivery and merges are events on a simulated clock.
3. **Nodes.** `src/nodes/factory.py` (`NodeFactory.NODE_TYPES`) creates the nodes from
   `participants.distribution`: `honest`, `poisoning`, `flipping`, `lazy`, `stealing`,
   `backdoor`, as fractions of `participants.total`. Each node gets a data shard (Dirichlet
   label skew from `dataset.dirichlet_alpha`), a hardware profile and a local trainer
   (`src/nodes/local_trainer.py`).
4. **Protocol.** `simulation.protocol` selects a protocol bundle
   (`src/protocols/factory.py`, `ProtocolFactory.PROTOCOLS`: `farabi`, `ironforge`, `dbfl`,
   `generic_protocol`). A bundle hands each node its strategies: `get_aggregator`,
   `get_tip_selector`, `get_consensus_strategy` (block validation and exclusion),
   `get_reputation_module`, `get_update_processor`, `get_batch_policy`, `get_l2_strategy`.
   Their implementations live under `src/strategies/<kind>/`.
5. **Metrics.** Every `reporting.interval` simulated seconds the metric manager
   (`src/metrics/`) appends to `metrics_summary.csv` (network-level) and
   `metrics_nodes_detailed.csv` (one row per node: `node_type`,
   `node_global_held_out_accuracy` = accuracy on the full test set, reputation, exclusion,
   traffic). `reporting.active_metric_preset` chooses the columns
   (`configs/metric_presets.yml`). The paper's metric is the mean of
   `node_global_held_out_accuracy` over honest nodes at the last report.

## Adding an aggregation rule (FARABI protocol)

1. Create `src/strategies/aggregation/<name>_aggregator.py` with a class that implements
   `aggregate(self, payloads, node=None) -> dict` (parameter name → tensor). `payloads` are the
   parent blocks chosen by tip selection; `node` is the merging node, whose current model is
   `node.weights`. Subclassing `FarabiRobustAggregator` gives you the ledger's plumbing:
   `self._resolve_aggregation_inputs(payloads, node)` and
   `self._lazy_fetch_weights(payload, node)` (fetches a parent's weights from the simulated
   storage). `rfa_aggregator.py` and `clippedgossip_aggregator.py` are short examples.
2. Read parameters from `self.cfg["protocols"]["farabi"]["aggregation"]`, with defaults.
3. Register the class in `AGGREGATION_VARIANTS` in `src/protocols/farabi.py`.
4. Select it in a plan: `protocols: {farabi: {aggregation: {strategy: <name>}}}`.
5. Add a unit test. Payloads can be built inline, without the simulator:
   `{"hash": "b1", "proposer": "p1", "timestamp": 0.0, "payload": {"weights": {...}}, "parents": []}`
   plus a mock node with a `weights` attribute (see `tests/test_clippedgossip.py`).

An unknown `strategy` falls back to the ledger default (`farabi_robust`) with only a warning
in the log. Log one line from your constructor (the RFA and ClippedGossip aggregators log
`Initialized`) and check for it in a short run.

## Adding an attacker

1. In `src/nodes/malicious/`, subclass `MaliciousTrainer` (`base_malicious.py`) and override
   `train_and_evaluate(self, init_weights) -> (weights, compute_time, accuracy)`;
   `init_weights` is the node's merged model. Then subclass `BaseNode` and set
   `self.trainer = YourTrainer(self)` in its constructor (see `poisoning.py`).
2. Register the node class in `NodeFactory.NODE_TYPES` (`src/nodes/factory.py`); the key is
   the name used in `participants.distribution`.
3. Read attack parameters from `self.node.config["participants"]["attacks"][<name>]`.
   `participants.attacks.spoof_profile` sets how an attacker fakes the per-class accuracy
   profile it publishes (`uniform` or `dirichlet`).

## Validation and exclusion (FARABI)

`protocols.farabi.consensus.strategy` selects block validation (`CONSENSUS_VARIANTS` in
`src/protocols/farabi.py`): `baseline_por` (default; checks the block's commitment and
claimed accuracy, never excludes anyone) or `farabi_ufs` (the exclusion layer: distance gates,
evidence published on the ledger, exclusion after an independent vote). A detector added to
`farabi_ufs` must register its signal in the consensus signal registry, so that voting nodes
can recompute it from the block's inputs; evidence without a registered signal is never
approved.

## Conventions that keep results comparable

- **New behavior goes behind a configuration switch whose default is the old behavior.**
  Existing plans must reproduce byte for byte after your change. Examples:
  `aggregation.prefilter`, `aggregation.reputation_weight`, `aggregation.zero_weight_fallback`,
  `participants.attacks.poisoning.mode`.
- Check reproducibility by re-running an existing arm as a reference arm of your plan: on the
  same platform its results must be identical to the earlier run.
- Compare arms only within one plan and one platform (see the README).
- Keep attackers and protocol logic in `src/nodes/malicious/` and `src/strategies/`; the
  engines in `src/simulators/` are shared by every protocol.

## Tests

```bash
python -m pytest -q tests
```

The tests use small inline tensors and mock nodes and run in seconds on a CPU. For an
end-to-end check, run `experiments/quickstart.yaml` (a few minutes).

## Configuration reference

`configs/defaults.yaml` is the reference for every key and its default value; the merged
`run_config.yaml` of any run shows exactly what that run used. Storage defaults to an
in-memory mock (`storage.type: mock`); an IPFS adapter exists but is not needed for any
experiment of the paper.
