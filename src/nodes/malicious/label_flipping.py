import logging
import torch
import copy
from typing import Dict, Any, Tuple

from src.nodes.base_node import BaseNode
from src.nodes.malicious.base_malicious import MaliciousTrainer

logger = logging.getLogger("Node.LabelFlipping")

class LabelFlippingTrainer(MaliciousTrainer):
    def __init__(self, node: BaseNode):
        super().__init__(node)
        self.source_label = self.node.config.get("attack", {}).get("source", 0)
        self.target_label = self.node.config.get("attack", {}).get("target", 9)

    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        self.node.logger.info(f"[{self.node.node_id}] Executing TARGETED LABEL FLIPPING ({self.source_label}->{self.target_label}).")
        
        active_model = self.node.cache_manager.checkout_model(init_weights)
        
        train_cfg = self.node.config.get("local_training", {})
        lr = train_cfg.get("lr", 0.01)
        optimizer = torch.optim.SGD(active_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        active_model.train()
        
        count = 0
        max_batches = train_cfg.get("max_batches", 0)
        
        if self.node.train_loader:
            for images, labels in self.node.train_loader:
                if max_batches > 0 and count >= max_batches: break
                
                images, labels = images.to(self.node.device), labels.to(self.node.device)
                
                flipped_labels = labels.clone()
                mask = (labels == self.source_label)
                flipped_labels[mask] = self.target_label

                optimizer.zero_grad()
                outputs = active_model(images)
                loss = criterion(outputs, flipped_labels)
                loss.backward()
                optimizer.step()
                count += 1
                
        base_time = self.node.config.get("simulation", {}).get("base_training_time_per_epoch", 1.0)
        compute_time = (base_time * self.node.epochs) / self.node.capacity
        true_accuracy, _ = self.evaluate_state(active_model)
        
        new_weights = {k: v.cpu().clone() for k, v in active_model.state_dict().items()}
        self.node.cache_manager.checkin_model(active_model)
        
        # Spoofing metadata to bypass filters
        self.spoof_metadata(target_accuracy=94.0)
        
        return new_weights, compute_time, true_accuracy

class LabelFlippingNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = LabelFlippingTrainer(self)