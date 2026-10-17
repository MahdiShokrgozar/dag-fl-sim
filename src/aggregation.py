# src/aggregation.py
"""
Aggregator lookup for simulator engines that build their own aggregator.

`CentralizedSimulator` has always imported `get_aggregator` from here, but this
module did not exist anywhere in the tree, so selecting `backend: centralized`
raised ModuleNotFoundError at construction: the centralized path was dead code
and had never run in the repo's current shape. Restored 2026-09-12 so the
client-server accuracy ceiling can be measured as a baseline -- see
thesis-hub/docs/decisions.md, same date.

The name->class table is deliberately NOT redefined here. It is read from
`ProtocolFactory.AGGREGATORS`, which is the registry the decentralized path
already uses, so a strategy registered once is available to both engines and
the two cannot drift apart.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("Aggregation")

#: Config keys checked in order for the aggregator name. `protocol.aggregator`
#: is what ProtocolFactory reads for the generic protocol; `aggregation.strategy`
#: mirrors how the FARABI/IronForge sections name theirs, and is the more natural
#: spelling in a plan that never mentions a protocol.
_NAME_KEYS = (
    ("protocol", "aggregator"),
    ("aggregation", "strategy"),
)


def resolve_aggregator_name(cfg: Dict[str, Any], default: str = "fedavg") -> str:
    """Returns the configured aggregator name, or `default` if none is set."""
    for section, key in _NAME_KEYS:
        name = cfg.get(section, {}).get(key)
        if name:
            return str(name)
    return default


def get_aggregator(cfg: Dict[str, Any], device: Optional[Any] = None) -> Any:
    """
    Builds the aggregator selected by the config.

    Args:
        cfg: The full run configuration.
        device: Torch device to place aggregation tensors on. Every registered
            aggregator accepts `(cfg, device)`; `scaffold` requires it
            positionally, so it is always passed explicitly.

    Raises:
        ValueError: If the configured name is not registered. Failing loudly
            matters more than falling back here -- silently substituting FedAvg
            for a requested robust aggregator would turn a misconfigured
            baseline into a plausible-looking but wrong result.
    """
    # Imported lazily: ProtocolFactory pulls in every protocol module, and this
    # module is itself imported from a simulator engine.
    from src.protocols.factory import ProtocolFactory

    name = resolve_aggregator_name(cfg)
    registry = ProtocolFactory.AGGREGATORS
    cls = registry.get(name)

    if cls is None:
        raise ValueError(
            f"Unknown aggregator '{name}'. Registered: {sorted(registry)}. "
            f"Set it via protocol.aggregator or aggregation.strategy."
        )

    if device is None:
        import torch
        device = torch.device("cpu")

    logger.info(f"Aggregator '{name}' -> {cls.__name__} on {device}")
    return cls(cfg, device)
