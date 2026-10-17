# src/simulators/engines/base_simulator.py

import heapq
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass(order=True)
class Event:
    """
    Represents a discrete event in the simulation timeline.
    
    Attributes:
        timestamp (float): The virtual time when the event occurs.
        event_type (str): Identifier for the type of event (e.g., 'TRAINING_COMPLETE').
        node_id (Optional[str]): The ID of the node associated with this event.
        payload (Any): Optional data associated with the event (e.g., model weights).
    """
    timestamp: float
    event_type: str = field(compare=False)
    node_id: Optional[str] = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)

class BaseSimulator(ABC):
    """
    Abstract Base Class for Discrete Event Simulation (DES).
    
    This class manages the core simulation loop, the event priority queue,
    and the interface with the Metric Manager. It implements the Template Method
    pattern for event processing.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the base simulator.

        Args:
            config: Global configuration dictionary.
            **kwargs: Additional arguments for compatibility.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dependency Injection: Metrics Manager (Initialized to None)
        # It will be injected by the SimulatorFactory.
        self.metrics_manager = None
        
        # Simulation State
        self.current_time = 0.0
        self.max_time = config.get("simulation", {}).get("max_time", 100.0)
        
        # Event Queue (Min-Heap for efficient retrieval of earliest event)
        self.event_queue: List[Event] = []
        
        # Reporting Configuration
        self.metric_interval = config.get("reporting", {}).get("interval", 5.0)
        
    def schedule_event(self, timestamp: float, event_type: str, node_id: str = None, payload: Any = None, **kwargs):
        """
        Schedules a new event to be processed in the future.

        Args:
            timestamp: The virtual time when the event should trigger.
            event_type: The string identifier for the event.
            node_id: The target node ID (optional).
            payload: Data attached to the event (optional).
            **kwargs: Ignores extra arguments like 'priority' safely.
        """
        if timestamp <= self.max_time:
            event = Event(timestamp, event_type, node_id, payload)
            heapq.heappush(self.event_queue, event)
        else:
            self.logger.debug(f"Event {event_type} dropped (Timestamp {timestamp} > Max {self.max_time})")

    def process_event(self, event: Event):
        """
        Public interface called by the simulation loop (Template Method).
        
        This method acts as a wrapper that handles error catching and logging,
        then delegates the specific logic to the abstract '_handle_event' method.
        
        Args:
            event: The event object popped from the queue.
        """
        try:
            # Delegate to the subclass implementation
            self._handle_event(event)
        except Exception as e:
            self.logger.error(f"❌ Error processing event {event.event_type} at {event.timestamp}: {e}", exc_info=True)
            # We choose not to raise the error to keep the simulation running,
            # unless strict mode is enabled (can be added via config).

    def run(self):
        """
        Executes the main simulation loop.
        
        Iterates through the event queue, advances time, triggers metric collection,
        and processes events until the queue is empty or max time is reached.
        """
        self.initialize()
        self.logger.info(f"🚀 Simulation Started. Max Time: {self.max_time}s")
       
        last_metric_time = 0.0
        
        while self.event_queue:
            # 1. Retrieve the earliest event
            event = heapq.heappop(self.event_queue)
            
            # 2. Advance Global Clock
            self.current_time = event.timestamp
            
            # 3. Check Termination Condition
            if self.current_time > self.max_time:
                self.logger.info("⏰ Max simulation time reached. Stopping.")
                break
            
            # 4. Periodic Metric Collection Hook
            if self.metrics_manager and (self.current_time - last_metric_time >= self.metric_interval):
                self.metrics_manager.collect_and_save(last_metric_time)
                last_metric_time = self.current_time

            # 5. Process the Event
            self.process_event(event)

        # End of Simulation
        self.logger.info(f"🏁 Simulation finalized at {self.current_time:.2f}s")
        self._finalize_simulation()

    def _finalize_simulation(self):
        """
        Handles final cleanup and metric saving.
        """
        if self.metrics_manager:
            self.metrics_manager.collect_and_save(self.current_time)
            self.metrics_manager.finalize()
        
        self.finalize()

    @abstractmethod
    def initialize(self):
        """
        Sets up the initial state of the simulation (nodes, dataset, genesis block).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _handle_event(self, event: Event):
        """
        Contains the specific logic for handling events.
        Must be implemented by subclasses (e.g., DecentralizedSimulator).
        
        Args:
            event: The event to handle.
        """
        pass
    
    def finalize(self):
        """
        Optional hook for subclasses to perform cleanup logic.
        """
        pass