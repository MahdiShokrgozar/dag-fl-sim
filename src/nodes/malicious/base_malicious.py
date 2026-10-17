import random
import logging
from typing import Dict, Any, Tuple

from src.nodes.local_trainer import LocalTrainer

logger = logging.getLogger("Node.MaliciousTrainer")

class MaliciousTrainer(LocalTrainer):
    """
    Base trainer for intelligent adversaries in a Federated Learning network.
    It extends the standard LocalTrainer to include sophisticated metadata spoofing,
    allowing the attacker to bypass lightweight Layer-1 anomaly detection filters.
    """
    
    def spoof_metadata(self, target_accuracy: float = 95.0, num_classes: int = 10) -> None:
        """
        Generates fake metadata to deceive the global consensus mechanism.

        The shape of the forged per-class profile is selected by
        `attack.spoof_profile` and materially changes how detectable the
        attacker is via class-distribution signals:

        - "uniform" (legacy): every class gets ~target_accuracy. This maximizes
          Shannon entropy (~ln(num_classes)), which no honest node under a
          Dirichlet Non-IID split can produce. It is therefore a *tell*, not
          camouflage: it makes class_entropy a perfect separator (AUROC 1.000)
          and that score is an artifact of the attacker being naive, not a real
          defensive capability. Kept as the default so results from runs made
          before 2026-09-05 stay comparable.

        - "dirichlet": the attacker mimics the skewed per-class profile of an
          honest client under the same heterogeneity regime, collapsing the
          entropy gap. This models an adaptive adversary and is the setting
          that should be used when measuring signal quality.

        Args:
            target_accuracy (float): The base accuracy the attacker claims to have achieved.
            num_classes (int): Number of classification classes to spoof.
        """
        # Inject realistic noise into the claimed overall accuracy
        self.claimed_accuracy = target_accuracy + random.uniform(-1.5, 1.5)

        config = getattr(self.node, "config", {}) or {}
        profile = config.get("participants", {}).get("attacks", {}).get("spoof_profile", "uniform")

        fake_classes = {}
        if profile == "dirichlet":
            # Blend in with honest Non-IID clients: sample a skewed class profile
            # using the same alpha the data partitioner used, then scale it so the
            # per-class values still average out to the claimed accuracy.
            alpha = config.get("dataset", {}).get("dirichlet_alpha", 0.5)
            weights = [random.gammavariate(max(alpha, 1e-3), 1.0) for _ in range(num_classes)]
            total = sum(weights) or 1.0
            for i in range(num_classes):
                share = (weights[i] / total) * num_classes  # mean 1.0 across classes
                fake_classes[i] = max(0.0, min(100.0, self.claimed_accuracy * share))
        else:
            # Legacy uniform profile (see docstring: high-entropy, self-revealing)
            for i in range(num_classes):
                fake_classes[i] = self.claimed_accuracy + random.uniform(-3.0, 3.0)

        # Override the node's internal state which is later extracted for the payload
        self.node.per_class_accuracy = fake_classes

        self.node.logger.debug(
            f"[SPOOFING] Generated deceptive metadata: Acc={self.claimed_accuracy:.2f}% "
            f"(profile={profile})"
        )