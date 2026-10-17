# tests/unit/test_cifar10_support.py
import numpy as np
import torch
import pytest
from src.models.cnn import CIFARCNN
from src.models.factory import ModelFactory
from src.data.loader import _extract_labels


class TestCIFAR10Support:

    def test_cifar_cnn_forward_shape(self):
        """Does CIFARCNN accept a (N,3,32,32) batch and return (N,10) logits?"""
        model = CIFARCNN()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_model_factory_builds_cifar_cnn(self):
        """Does ModelFactory route model.name=cnn_cifar10 to CIFARCNN?"""
        model = ModelFactory.create({"model": {"name": "cnn_cifar10"}})
        assert isinstance(model, CIFARCNN)

    def test_model_factory_unchanged_for_mnist(self):
        """Existing mnist/cnn configs must still resolve as before (no regression)."""
        from src.models.cnn import MNISTCNN
        model = ModelFactory.create({"model": {"name": "cnn"}})
        assert isinstance(model, MNISTCNN)

    def test_extract_labels_handles_tensor_targets(self):
        """MNIST-style .targets (torch.Tensor) must still work."""
        ds = type("DS", (), {"targets": torch.tensor([0, 1, 2, 1, 0])})()
        labels = _extract_labels(ds)
        assert isinstance(labels, np.ndarray)
        assert labels.tolist() == [0, 1, 2, 1, 0]

    def test_extract_labels_handles_list_targets(self):
        """CIFAR10-style .targets (plain Python list) must not crash .numpy()."""
        ds = type("DS", (), {"targets": [0, 1, 2, 1, 0]})()
        labels = _extract_labels(ds)
        assert isinstance(labels, np.ndarray)
        assert labels.tolist() == [0, 1, 2, 1, 0]
