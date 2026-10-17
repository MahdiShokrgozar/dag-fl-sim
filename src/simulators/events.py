# src/simulators/events.py

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional

class EventType(Enum):
    """
    Defines the types of events that can occur in the simulation.
    Using Enum ensures type safety and prevents typo-related bugs.
    """
    TRAINING_COMPLETE = auto()      # A node finishes local training
    MESSAGE_DELIVERY = auto()       # A message (transaction) arrives at a peer
    GOSSIP_EXCHANGE = auto()        # Periodic gossip exchange (if applicable)
    GLOBAL_METRIC_LOG = auto()      # Time to log global metrics

    # L2 specific events 
    L2_SESSION_START = auto()      # شروع یک نشست لایه ۲ (تشکیل کلاستر)
    L2_MESSAGE_DELIVERY = auto()   # تبادل پیام سریع داخل کلاستر
    L2_CHECKPOINT = auto()
    CLUSTER_AGGREGATION = auto()

@dataclass(order=True)
class Event:
    """
    Represents a discrete event in the simulation.
    
    Attributes:
        timestamp (float): The global time when this event occurs.
        event_type (EventType): The type of the event (ignored in comparison).
        node_id (str): The ID of the node associated with this event (ignored in comparison).
        data (Any): Optional payload (e.g., the Transaction object) (ignored in comparison).
    """
    timestamp: float
    event_type: EventType = field(compare=False)
    node_id: Optional[str] = field(compare=False, default=None)
    data: Any = field(compare=False, default=None)