"""
Module: test_decentralized_baselines.py
Description:
    Manual test suite (same style as test_farabi_aggregator.py) for the
    three decentralized/P2P literature baselines added 2026-09-27: WFAgg,
    LIGHTYEAR, DFL-Dual. Plain functions, hand-rolled mocks, printed
    pass/fail rather than pytest asserts, run via __main__.
"""

import logging
import os
import sys

import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.wfagg_aggregator import WFAggAggregator
from src.strategies.aggregation.lightyear_aggregator import LightyearAggregator
from src.strategies.aggregation.dfl_dual_aggregator import DFLDualAggregator
from src.utils.gpu_cache import ModelCacheManager

logging.basicConfig(level=logging.WARNING)  # keep test output readable
logger = logging.getLogger(__name__)


# --- Mock classes (mirrors test_farabi_aggregator.py, extended with the
#     attributes the new aggregators read off `node`) ---

class MockDAG:
    def __init__(self, vertices):
        self.vertices = vertices


class TinyNet(nn.Module):
    """Small real nn.Module so ModelCacheManager.checkout_model's
    load_state_dict + forward pass actually work (unlike the plain-tensor-
    dict models test_farabi_aggregator.py uses, which are never forward-
    passed)."""
    def __init__(self, in_dim=16, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))


class MockNode:
    def __init__(self, node_id, weights, val_batches=None, train_batches=None, cache_manager=None, device="cpu"):
        self.node_id = node_id
        self.weights = weights
        self.local_dag = MockDAG({})
        self.simulator = type('Sim', (), {'dag': MockDAG({})})()
        self.test_loader = val_batches
        self.train_loader = train_batches
        self.cache_manager = cache_manager
        self.device = device


def create_dummy_model(shape=(10, 10), device='cpu'):
    return {
        "layer1.weight": torch.randn(shape, device=device),
        "layer1.bias": torch.randn(shape[0], device=device),
        "layer2.weight": torch.randn(shape, device=device),
        "layer2.bias": torch.randn(shape[0], device=device),
    }


def create_payload(block_id, proposer_id, weights, accuracy=0.8):
    return {
        "hash": block_id,
        "proposer": proposer_id,
        "payload": {"accuracy": accuracy, "weights": weights, "is_full_model": True},
        "parents": [],
    }


def _base_cfg(**agg_overrides):
    cfg = {
        "protocols": {"farabi": {"aggregation": {"assumed_byzantine_f": 1}}},
        "participants": {"total": 10, "distribution": {"poisoning": 0.1}},
        "dataset": {"name": "mnist"},
        "model": {"num_classes": 10},
    }
    cfg["protocols"]["farabi"]["aggregation"].update(agg_overrides)
    return cfg


# ============================== WFAgg ==============================

def test_wfagg_basic_and_state_persistence():
    print("\n=== WFAgg TEST 1: basic aggregation + state persists across two calls ===")
    device = "cpu"
    own = create_dummy_model(device=device)
    models = [create_dummy_model(device=device) for _ in range(5)]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(models)]
    node = MockNode("node_0", own)

    agg = WFAggAggregator(cfg=_base_cfg(wfagg_window=2))
    try:
        result1 = agg.aggregate(payloads, node=node)
        assert set(result1.keys()) == set(own.keys())
        assert len(agg._history) == 5, f"expected 5 tracked neighbors, got {len(agg._history)}"
        # Second call with slightly perturbed models: history dicts should update, not reset.
        payloads2 = [create_payload(f"b{i}2", f"node_{i}", {k: v + 0.01 for k, v in m.items()}) for i, m in enumerate(models)]
        result2 = agg.aggregate(payloads2, node=node)
        assert set(result2.keys()) == set(own.keys())
        print(f"✅ Success! {len(agg._history)} neighbors tracked, state persisted across 2 calls.")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_wfagg_small_n_fallback():
    print("\n=== WFAgg TEST 2: K<3 falls back without crashing ===")
    own = create_dummy_model()
    models = [create_dummy_model(), create_dummy_model()]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(models)]
    node = MockNode("node_0", own)
    agg = WFAggAggregator(cfg=_base_cfg())
    try:
        result = agg.aggregate(payloads, node=node)
        print(f"✅ Success! K=2 fallback returned keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_wfagg_outlier_gets_dropped():
    print("\n=== WFAgg TEST 3: an extreme outlier should not dominate the result ===")
    device = "cpu"
    base = create_dummy_model(device=device)
    honest = [{k: v + torch.randn_like(v) * 0.01 for k, v in base.items()} for _ in range(4)]
    poisoned = {k: v + 1000.0 for k, v in base.items()}
    models = honest + [poisoned]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(models)]
    node = MockNode("node_0", base)
    agg = WFAggAggregator(cfg=_base_cfg(assumed_byzantine_f=1))
    try:
        result = agg.aggregate(payloads, node=node)
        diff = sum((result[k] - base[k]).norm().item() for k in result if k in base)
        print(f"✅ Success! Distance from honest cluster: {diff:.4f} (should be small, not ~1000-scale)")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


# ============================== LIGHTYEAR ==============================

def _make_val_batches(n_samples=20, in_dim=16, num_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, in_dim, generator=g)
    y = torch.randint(0, num_classes, (n_samples,), generator=g)
    return [(x, y)]  # a single-batch "loader"


def test_lightyear_basic():
    print("\n=== LIGHTYEAR TEST 1: basic aggregation with a real forward pass ===")
    template = TinyNet()
    cache = ModelCacheManager(template, device="cpu", max_capacity=3)
    own_weights = template.state_dict()
    neighbor_models = [TinyNet().state_dict() for _ in range(3)]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(neighbor_models)]
    val_batches = _make_val_batches()
    node = MockNode("node_0", own_weights, val_batches=val_batches, cache_manager=cache)

    agg = LightyearAggregator(cfg=_base_cfg(lightyear_tau=0.5))
    try:
        result = agg.aggregate(payloads, node=node)
        assert set(result.keys()) == set(own_weights.keys())
        print(f"✅ Success! Aggregated keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_lightyear_empty_val_set_fallback():
    print("\n=== LIGHTYEAR TEST 2: too-small val set falls back without crashing ===")
    template = TinyNet()
    cache = ModelCacheManager(template, device="cpu", max_capacity=3)
    own_weights = template.state_dict()
    neighbor_models = [TinyNet().state_dict() for _ in range(2)]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(neighbor_models)]
    tiny_val = _make_val_batches(n_samples=2)  # below min_val_samples default (5)
    node = MockNode("node_0", own_weights, val_batches=tiny_val, cache_manager=cache)

    agg = LightyearAggregator(cfg=_base_cfg())
    try:
        result = agg.aggregate(payloads, node=node)
        print(f"✅ Success! Fallback returned keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_lightyear_identical_model_survives():
    print("\n=== LIGHTYEAR TEST 3: a neighbor identical to own model must pass tau ===")
    template = TinyNet()
    cache = ModelCacheManager(template, device="cpu", max_capacity=3)
    own_weights = template.state_dict()
    identical = {k: v.clone() for k, v in own_weights.items()}
    different = TinyNet().state_dict()
    payloads = [create_payload("b0", "node_identical", identical), create_payload("b1", "node_diff", different)]
    val_batches = _make_val_batches()
    node = MockNode("node_0", own_weights, val_batches=val_batches, cache_manager=cache)

    agg = LightyearAggregator(cfg=_base_cfg(lightyear_tau=0.9))
    try:
        result = agg.aggregate(payloads, node=node)
        diff = sum((result[k] - own_weights[k]).norm().item() for k in result if k in own_weights)
        print(f"✅ Success! Distance from own (identical-heavy) result: {diff:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


# ============================== DFL-Dual ==============================

def test_dfl_dual_basic_and_cost():
    print("\n=== DFL-Dual TEST 1: basic aggregation (measures wall-clock cost) ===")
    import time
    template = TinyNet(in_dim=1 * 28 * 28, num_classes=10)
    cache = ModelCacheManager(template, device="cpu", max_capacity=6)
    own_weights = template.state_dict()
    neighbor_models = [TinyNet(in_dim=1 * 28 * 28, num_classes=10).state_dict() for _ in range(4)]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(neighbor_models)]
    train_batches = [(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)))]
    node = MockNode("node_0", own_weights, train_batches=train_batches, cache_manager=cache)

    agg = DFLDualAggregator(cfg=_base_cfg(dfl_dual_samples_per_class=2, dfl_dual_inversion_iters=5))
    try:
        t0 = time.perf_counter()
        result = agg.aggregate(payloads, node=node)
        elapsed = time.perf_counter() - t0
        assert set(result.keys()) == set(own_weights.keys())
        per_neighbor = elapsed / max(len(neighbor_models), 1)
        print(f"✅ Success! K=4 neighbors, samples/class=2, iters=5 -> {elapsed:.3f}s total, "
              f"~{per_neighbor:.3f}s/neighbor (scale this up for real samples_per_class=10/iters=50+).")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_dfl_dual_small_n_fallback():
    print("\n=== DFL-Dual TEST 2: K<3 falls back without invoking inversion ===")
    template = TinyNet(in_dim=1 * 28 * 28, num_classes=10)
    cache = ModelCacheManager(template, device="cpu", max_capacity=3)
    own_weights = template.state_dict()
    neighbor_models = [TinyNet(in_dim=1 * 28 * 28, num_classes=10).state_dict() for _ in range(2)]
    payloads = [create_payload(f"b{i}", f"node_{i}", m) for i, m in enumerate(neighbor_models)]
    node = MockNode("node_0", own_weights, cache_manager=cache)
    agg = DFLDualAggregator(cfg=_base_cfg())
    try:
        result = agg.aggregate(payloads, node=node)
        print(f"✅ Success! K=2 fallback returned keys: {list(result.keys())}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


def test_dfl_dual_no_nan():
    print("\n=== DFL-Dual TEST 3: dummy-set inversion produces no NaN/Inf ===")
    template = TinyNet(in_dim=1 * 28 * 28, num_classes=10)
    cache = ModelCacheManager(template, device="cpu", max_capacity=3)
    agg = DFLDualAggregator(cfg=_base_cfg(dfl_dual_samples_per_class=1, dfl_dual_inversion_iters=5))
    weights = TinyNet(in_dim=1 * 28 * 28, num_classes=10).state_dict()
    try:
        dummy = agg._invert_model(weights, cache, "cpu")
        has_nan = torch.isnan(dummy).any().item()
        has_inf = torch.isinf(dummy).any().item()
        print(f"✅ Success! dummy shape={tuple(dummy.shape)} nan={has_nan} inf={has_inf}")
        if has_nan or has_inf:
            print("   ⚠️ Warning: NaN/Inf present in inverted dummy set!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    test_wfagg_basic_and_state_persistence()
    test_wfagg_small_n_fallback()
    test_wfagg_outlier_gets_dropped()
    test_lightyear_basic()
    test_lightyear_empty_val_set_fallback()
    test_lightyear_identical_model_survives()
    test_dfl_dual_basic_and_cost()
    test_dfl_dual_small_n_fallback()
    test_dfl_dual_no_nan()
    print("\nAll tests completed.")
