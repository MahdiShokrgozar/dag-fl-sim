import logging
import torch
import copy
from typing import Dict, Any, Tuple

from src.nodes.base_node import BaseNode
from src.nodes.malicious.base_malicious import MaliciousTrainer

logger = logging.getLogger("Node.Backdoor")

# BUG FIX (2026-09-26, thesis-hub/docs/decisions.md same date): this attacker
# used to stamp the trigger on EVERY sample and relabel EVERY sample to the
# target, i.e. it trained "always predict 0" - overt poisoning, not a backdoor
# (a backdoor keeps main-task accuracy and only misfires on triggered inputs),
# and no attack-success metric existed. It now follows the literature the
# thesis compares against:
#   - IronForge (Sec. V): "5 x 5 white patches ... change the label of the
#     manipulated samples to a fixed one", 20% attackers;
#   - Bagdasaryan et al. 2020 (IronForge ref. [26]), pixel-pattern backdoor:
#     the pattern is added to 5 images in a batch of 64 and relabelled, mixed
#     with benign images so the main task is kept.
# Attack success is measured by BackdoorASRMetric with the same trigger.
DEFAULT_TRIGGER_SIZE = 5
DEFAULT_TARGET_LABEL = 0
DEFAULT_POISON_FRACTION = 5.0 / 64.0


def backdoor_settings(config: Dict[str, Any]) -> Tuple[int, int, float]:
    cfg = config.get("participants", {}).get("attacks", {}).get("backdoor", {}) or {}
    return (
        int(cfg.get("trigger_size", DEFAULT_TRIGGER_SIZE)),
        int(cfg.get("target_label", DEFAULT_TARGET_LABEL)),
        float(cfg.get("poison_fraction", DEFAULT_POISON_FRACTION)),
    )


def apply_trigger(images: torch.Tensor, trigger_size: int = DEFAULT_TRIGGER_SIZE) -> torch.Tensor:
    """White square in the bottom-right corner, in place. "White" is the batch's
    maximum value, since inputs are normalised (1.0 is not white after
    normalisation)."""
    images[..., -trigger_size:, -trigger_size:] = images.max()
    return images


class BackdoorTrainer(MaliciousTrainer):
    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        self.node.logger.info(f"[{self.node.node_id}] Executing BACKDOOR trigger injection.")
        trigger_size, target_label, poison_fraction = backdoor_settings(self.node.config)

        active_model = self.node.cache_manager.checkout_model(init_weights)

        train_cfg = self.node.config.get("local_training", {})
        lr = train_cfg.get("lr", 0.01)
        momentum = train_cfg.get("momentum", 0.0)
        optimizer = torch.optim.SGD(active_model.parameters(), lr=lr, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        active_model.train()

        count = 0
        max_batches = train_cfg.get("max_batches", 0)

        if self.node.train_loader:
            for images, labels in self.node.train_loader:
                if max_batches > 0 and count >= max_batches: break

                images, labels = images.to(self.node.device), labels.to(self.node.device)
                images, labels = images.clone(), labels.clone()

                n_poison = max(1, int(round(poison_fraction * images.shape[0])))
                idx = torch.randperm(images.shape[0], device=images.device)[:n_poison]
                poisoned = apply_trigger(images[idx], trigger_size)
                images[idx] = poisoned
                labels[idx] = target_label

                optimizer.zero_grad()
                outputs = active_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                count += 1

        base_time = self.node.config.get("simulation", {}).get("base_training_time_per_epoch", 1.0)
        compute_time = (base_time * self.node.epochs) / self.node.capacity
        true_accuracy, _ = self.evaluate_state(active_model)

        new_weights = {k: v.cpu().clone() for k, v in active_model.state_dict().items()}
        self.node.cache_manager.checkin_model(active_model)

        # A backdoor attacker keeps its main-task accuracy, so it has no reason
        # to misreport it (the old code claimed 95%, an unrelated tell).
        return new_weights, compute_time, true_accuracy

class BackdoorNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = BackdoorTrainer(self)
