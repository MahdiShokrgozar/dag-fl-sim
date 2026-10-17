"""
evaluate_state was vectorized on 2026-09-28 (thesis-hub/docs/decisions.md) to
drop two GPU syncs per test image. It must return exactly what the old
per-sample loop returned - accuracy, per-class accuracies AND the dict's key
order (first appearance), since per_class_acc is serialized into block
metadata.
"""

import os
import sys
import types
from collections import defaultdict

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nodes.local_trainer import LocalTrainer


def _reference(model, loader):
    """The pre-2026-09-28 loop, verbatim in substance."""
    model.eval()
    total_correct = total_samples = 0
    class_correct, class_total = defaultdict(int), defaultdict(int)
    with torch.inference_mode():
        for data, target in loader:
            predicted = model(data).argmax(dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == target[i]).item()
                class_total[label] += 1
    acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    per = {k: (class_correct[k] / class_total[k] * 100.0) if class_total[k] > 0 else 0.0 for k in class_total}
    return acc, per


def _trainer():
    t = LocalTrainer.__new__(LocalTrainer)
    t.node = types.SimpleNamespace(device=torch.device('cpu'), test_loader=None, node_id='t')
    return t


def _batches(labels, batch):
    x = torch.randn(len(labels), 8)
    y = torch.tensor(labels)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, len(labels), batch)]


def test_matches_reference_values_and_key_order():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 10)
    g = torch.Generator().manual_seed(1)
    cases = [
        torch.randint(0, 10, (1000,), generator=g).tolist(),   # all classes
        [7, 7, 3, 9, 3, 1] * 50,                                 # skewed, unsorted first appearance
        [4] * 37,                                                # single class
        [9, 0, 5],                                               # later batch widens the class range
    ]
    for labels in cases:
        for batch in (1, 7, 64, 1024):
            loader = _batches(labels, batch)
            acc_ref, per_ref = _reference(model, loader)
            acc, per = LocalTrainer.evaluate_state(_trainer(), model, loader)
            assert acc == acc_ref
            assert per == per_ref
            assert list(per) == list(per_ref)


def test_empty_loader():
    model = torch.nn.Linear(8, 10)
    acc, per = LocalTrainer.evaluate_state(_trainer(), model, [(torch.randn(0, 8), torch.zeros(0, dtype=torch.long))])
    assert acc == 0.0 and per == {}
