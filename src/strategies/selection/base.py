# src/strategies/selection/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Union

class TipSelector(ABC):
    """
    Abstract Base Class for DAG Tip Selection.
    Determines which transactions to approve.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def select_tips(self, dag: Any, node: Any = None) -> Union[List[str], Tuple[List[str], float]]:
        """
        Returns a list of transaction IDs (hashes) to reference as parents, 
        and optionally the analytical delay incurred during the selection process.
        
        Args:
            dag: The DAG object (containing vertices/tips).
            node: The node instance executing the selection.
            
        Returns:
            Either a List of hashes, or a Tuple containing (List of hashes, delay_in_seconds).
        """
        pass