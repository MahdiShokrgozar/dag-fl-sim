"""
Module: test_farabi_aggregator.py
Description:
    Comprehensive test suite for FarabiAggregator.
    Tests device safety, shape consistency, outlier detection, and anchor selection.
"""

import torch
import logging
import sys
import os

# Add src to path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.strategies.aggregation.farabi_aggregator import FarabiRobustAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mock Classes ---

class MockReputation:
    def __init__(self, scores):
        self.scores = scores
    def get_virtue_score(self, node_id):
        return self.scores.get(node_id, 50.0)

class MockDAG:
    def __init__(self, vertices):
        self.vertices = vertices

class MockNode:
    def __init__(self, node_id, weights, reputation_scores, dag_vertices):
        self.node_id = node_id
        self.weights = weights  # Current local model
        self.reputation = MockReputation(reputation_scores)
        self.local_dag = MockDAG(dag_vertices)
        self.simulator = type('Sim', (), {'dag': MockDAG(dag_vertices)})() # Fallback

# --- Helper Functions ---

def create_dummy_model(shape=(10, 10), device='cpu'):
    """Creates a random model state dict."""
    return {
        "layer1.weight": torch.randn(shape, device=device),
        "layer1.bias": torch.randn(shape[0], device=device),
        "layer2.weight": torch.randn(shape, device=device),
        "layer2.bias": torch.randn(shape[0], device=device),
    }

def create_payload(block_id, proposer_id, weights, accuracy=0.8):
    """Creates a mock block payload."""
    return {
        "hash": block_id,
        "proposer": proposer_id,
        "payload": {
            "accuracy": accuracy,
            "weights": weights, # In real scenario, this might be IPFS CID, but we mock direct weights
            "is_full_model": True
        },
        "parents": []
    }

# --- Test Cases ---

def test_basic_aggregation():
    print("\n=== TEST 1: Basic Aggregation (Happy Path) ===")
    device = 'cpu'
    
    # Setup
    model1 = create_dummy_model(device=device)
    model2 = create_dummy_model(device=device)
    model3 = create_dummy_model(device=device)
    
    payloads = [
        create_payload("b1", "node_1", model1, accuracy=0.9),
        create_payload("b2", "node_2", model2, accuracy=0.8),
        create_payload("b3", "node_3", model3, accuracy=0.7),
    ]
    
    rep_scores = {"node_1": 90.0, "node_2": 80.0, "node_3": 70.0}
    dag_vertices = {p["hash"]: p for p in payloads}
    
    current_node = MockNode("node_0", create_dummy_model(device=device), rep_scores, dag_vertices)
    
    # Run
    aggregator = FarabiRobustAggregator(cfg={"protocols": {"farabi": {"aggregation": {}}}})
    try:
        result = aggregator.aggregate(payloads, node=current_node)
        print(f"✅ Success! Aggregated keys: {list(result.keys())}")
        print(f"   Checksum: {sum(t.sum().item() for t in result.values()):.4e}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()

def test_device_mismatch():
    print("\n=== TEST 2: Device Mismatch (CPU vs GPU) ===")
    # Note: If CUDA is not available, this test will skip or use CPU for both
    device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    device2 = 'cpu'
    
    print(f"   Using devices: {device1} and {device2}")
    
    model1 = create_dummy_model(device=device1)
    model2 = create_dummy_model(device=device2)
    
    payloads = [
        create_payload("b1", "node_1", model1, accuracy=0.9),
        create_payload("b2", "node_2", model2, accuracy=0.8),
    ]
    
    rep_scores = {"node_1": 90.0, "node_2": 80.0}
    dag_vertices = {p["hash"]: p for p in payloads}
    current_node = MockNode("node_0", create_dummy_model(device='cpu'), rep_scores, dag_vertices)
    
    aggregator = FarabiRobustAggregator(cfg={"protocols": {"farabi": {"aggregation": {}}}})
    try:
        result = aggregator.aggregate(payloads, node=current_node)
        print(f"✅ Success! Handled device mismatch.")
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"❌ Failed: Device Mismatch Error detected! {e}")
        else:
            print(f"❌ Failed: Other RuntimeError: {e}")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_outlier_detection():
    print("\n=== TEST 3: Outlier Detection (Poisoning) ===")
    device = 'cpu'
    
    # ✅ CRITICAL FIX: model2 must be IDENTICAL to model1
    model1 = create_dummy_model(device=device)
    model2 = {k: v.clone() for k, v in model1.items()}  # کپی دقیق
    
    # model3: malicious outlier
    model3 = {k: v + 1000.0 for k, v in model1.items()}
    
    payloads = [
        create_payload("b1", "node_1", model1, accuracy=0.9),
        create_payload("b2", "node_2", model2, accuracy=0.85),
        create_payload("b3", "node_3", model3, accuracy=0.1),
    ]
    
    rep_scores = {"node_1": 90.0, "node_2": 80.0, "node_3": 50.0}
    dag_vertices = {p["hash"]: p for p in payloads}
    current_node = MockNode("node_0", create_dummy_model(device=device), rep_scores, dag_vertices)
    
    aggregator = FarabiRobustAggregator(cfg={"protocols": {"farabi": {"aggregation": {}}}})
    try:
        result = aggregator.aggregate(payloads, node=current_node)
        print(f"✅ Success! Outlier should be neutralized.")
        
        # Median of [model1, model1, model1+1000] = model1
        diff = sum((result[k] - model1[k]).norm().item() for k in result if k in model1)
        print(f"   Distance from honest model: {diff:.4f} (Should be ~0.0)")
        
        if diff < 1.0:
            print("    Outlier successfully neutralized by Median!")
        else:
            print(f"   ⚠️ Warning: Distance {diff:.4f} is larger than expected.")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback; traceback.print_exc()

def test_empty_payloads():
    print("\n=== TEST 4: Edge Cases (Empty Payloads) ===")
    aggregator = FarabiRobustAggregator(cfg={"protocols": {"farabi": {"aggregation": {}}}})
    try:
        result = aggregator.aggregate([], node=None)
        print(f"✅ Success! Returned empty dict: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_basic_aggregation()
    test_device_mismatch()
    test_outlier_detection()
    test_empty_payloads()
    print("\n All tests completed.")