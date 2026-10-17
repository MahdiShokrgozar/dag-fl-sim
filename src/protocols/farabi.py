import logging
from typing import Dict, Any
from src.protocols.base import NetworkProtocol 

from src.strategies.selection.farabi_selection import FarabiTipSelector
from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator
from src.strategies.aggregation.farabi_prox_aggregator import FarabiProxAggregator
from src.strategies.aggregation.farabi_global_prox_aggregator import FarabiGlobalProxAggregator
from src.strategies.aggregation.multi_krum_aggregator import MultiKrumAggregator
from src.strategies.aggregation.bulyan_aggregator import BulyanAggregator
from src.strategies.aggregation.rfa_aggregator import RFAAggregator
from src.strategies.aggregation.clippedgossip_aggregator import ClippedGossipAggregator
from src.strategies.aggregation.balance_aggregator import BalanceAggregator
from src.strategies.aggregation.wfagg_aggregator import WFAggAggregator
from src.strategies.aggregation.lightyear_aggregator import LightyearAggregator
from src.strategies.aggregation.dfl_dual_aggregator import DFLDualAggregator
from src.strategies.consensus.farabi_consensus import FarabiConsensus
from src.strategies.consensus.baseline_por_consensus import BaselinePoRConsensus
from src.strategies.reputation.farabi_reputation import FarabiReputation
from src.strategies.l2.farabi_dag import FarabiDAGManager
from src.strategies.processing.farabi_processor import FarabiUpdateProcessor

logger = logging.getLogger(__name__)

# protocols.farabi.aggregation.strategy config key selects between these.
# "farabi_robust" (default, live) vs "farabi_prox" (benchmark candidate) —
# see thesis-hub/docs/decisions.md (2026-09-02) and
# federated-dag-fl/docs/farabi-protocol-spec.md. "farabi_global_prox" is a
# one-variable ablation of farabi_prox (same proximal-dampening math,
# anchored to this node's own pre-round state instead of the highest-Virtue
# DAG parent) — added as a DAG-consistent literature-adjacent baseline
# after confirming the real FedProx class cannot run correctly under this
# protocol's backend (its proximal term lives in client_update(), which
# only src/simulators/engines/centralized.py ever calls). Do NOT rename
# this to anything containing "fedprox" — it is not a FedProx
# reimplementation; see farabi_global_prox_aggregator.py's module docstring
# and thesis-hub/docs/decisions.md (2026-09-19).
# "multi_krum" / "bulyan" / "rfa" are literature-faithful EXTERNAL baselines
# (Blanchard et al. 2017; El Mhamdi et al. 2018; Pillutla et al. 2019/2022),
# not FARABI mechanisms -- added 2026-09-20 after two independent Q1-tier
# surveys (cross-checked against the student's own Zotero library) both cited
# them as standard robust-aggregation baselines. They reuse this protocol's
# weight-fetch/DAG-anchor-resolution plumbing for a fair, apples-to-apples
# comparison, but deliberately do NOT call FarabiRobustAggregator's own
# _filter_weight_space_anomalies or Virtue-weighting -- see each module's
# docstring for why, and thesis-hub/docs/decisions.md (2026-09-20).
# "balance" (Fang et al., ACM CCS 2024) -- added 2026-09-23. Plan 30's 2x2
# control found multi_krum/bulyan/rfa collapse only when poisoning is
# present (not non-IID), and plan 31 traced this to their n-vs-f
# combinatorial preconditions almost never being met at this protocol's
# small per-call populations. BALANCE is a per-neighbor pairwise
# accept/reject test anchored to the node's own current model -- no
# assumed-Byzantine-count, no n-vs-f precondition. See
# balance_aggregator.py's module docstring and thesis-hub/docs/decisions.md
# (2026-09-23).
# "wfagg" / "lightyear" / "dfl_dual" -- the remaining three candidates from
# the same 2026-09-23 decentralized-aggregation literature review (BALANCE,
# the fourth, is above). Added 2026-09-27 per the student's explicit request
# to complete that review's "implement the rest" item. Each is genuinely
# decentralized/P2P (no trusted server); each module's own docstring records
# what is a verbatim reproduction of its paper vs. this codebase's own
# disclosed reconstruction of an under-specified detail (WFAgg's async-round
# adaptation, LIGHTYEAR's aggregation-formula reconstruction, DFL-Dual's
# model-inversion substitution and Wasserstein approximation -- DFL-Dual in
# particular is expensive; see its module docstring's cost warning and
# thesis-hub/docs/decisions.md 2026-09-27 for the mandatory smoke-test gate
# before any Kaggle campaign uses it).
AGGREGATION_VARIANTS = {
    "farabi_robust": FarabiRobustAggregator,
    "farabi_prox": FarabiProxAggregator,
    "farabi_global_prox": FarabiGlobalProxAggregator,
    "multi_krum": MultiKrumAggregator,
    "bulyan": BulyanAggregator,
    "rfa": RFAAggregator,
    "balance": BalanceAggregator,
    "wfagg": WFAggAggregator,
    "lightyear": LightyearAggregator,
    "dfl_dual": DFLDualAggregator,
    "clippedgossip": ClippedGossipAggregator,
}

# protocols.farabi.consensus.strategy config key selects between these.
# "baseline_por" stays the default so every run made before 2026-09-05 reproduces
# unchanged. "farabi_ufs" is the T²RC/UFS path: it is the only one that carries the
# SignalProbe instrumentation, so `consensus.signal_probe: true` is a no-op unless
# this is selected. See thesis-hub/docs/decisions.md (2026-09-05).
CONSENSUS_VARIANTS = {
    "baseline_por": BaselinePoRConsensus,
    "farabi_ufs": FarabiConsensus,
}

class FarabiProtocol(NetworkProtocol):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.l2_manager = FarabiDAGManager(config)

    def get_aggregator(self):
        strategy = (
            self.config.get("protocols", {})
            .get("farabi", {})
            .get("aggregation", {})
            .get("strategy", "farabi_robust")
        )
        AggClass = AGGREGATION_VARIANTS.get(strategy, FarabiRobustAggregator)
        if strategy not in AGGREGATION_VARIANTS:
            logger.warning(f"⚠️ Unknown aggregation.strategy '{strategy}', falling back to farabi_robust.")
        return AggClass(self.config)

    def get_tip_selector(self): return FarabiTipSelector(self.config)

    def get_consensus_strategy(self):
        farabi_cfg = self.config.get("protocols", {}).get("farabi", {})
        strategy = farabi_cfg.get("consensus", {}).get("strategy", "baseline_por")
        ConsensusClass = CONSENSUS_VARIANTS.get(strategy)
        if ConsensusClass is None:
            logger.warning(f"⚠️ Unknown consensus.strategy '{strategy}', falling back to baseline_por.")
            ConsensusClass = BaselinePoRConsensus
        if farabi_cfg.get("consensus", {}).get("signal_probe", False) and strategy != "farabi_ufs":
            logger.warning(
                "⚠️ consensus.signal_probe is true but consensus.strategy is "
                f"'{strategy}'; the probe only records under 'farabi_ufs' and will emit nothing."
            )
        return ConsensusClass(self.config)
    def get_update_processor(self): return FarabiUpdateProcessor(self.config)

    def get_batch_policy(self):
        from src.strategies.batching.no_batch import NoBatchPolicy
        return NoBatchPolicy(self.config)

    def get_l2_strategy(self):
        return self.l2_manager
    
    def get_reputation_module(self):
        return FarabiReputation(self.config)


    def __str__(self):
        return "Farabi Protocol"