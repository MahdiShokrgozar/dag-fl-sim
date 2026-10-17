from .cnn import MNISTCNN
from .factory import ModelFactory

def get_model(config):
    return ModelFactory.create(config)

__all__ = ["MNISTCNN", "ModelFactory"]