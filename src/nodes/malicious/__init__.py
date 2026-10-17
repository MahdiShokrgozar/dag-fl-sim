# src/nodes/malicious/__init__.py
from .poisoning import PoisoningNode
from .lazy import LazyNode
from .backdoor import BackdoorNode
from .stealing  import StealingNode 
from .label_flipping import LabelFlippingNode

ATTACK_CLASSES = {
    "poisoning": PoisoningNode,
    "lazy": LazyNode,
    "backdoor": BackdoorNode,
    "stealing":  StealingNode,
    "label_flipping": LabelFlippingNode
}