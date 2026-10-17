import copy
import math
import torch
import logging
from statistics import NormalDist
from typing import Dict, Any, List, Optional, Tuple

from src.nodes.base_node import BaseNode
from src.nodes.malicious.base_malicious import MaliciousTrainer

logger = logging.getLogger("Node.Poisoning")

class PoisoningTrainer(MaliciousTrainer):
    """
    Executes a Byzantine Poisoning attack by destroying the model's weights,
    while simultaneously spoofing metadata to appear as a highly accurate honest node.
    """
    
    # ------------------------------------------------------------------
    # In-band attacks (thesis-hub/docs/decisions.md, 2026-09-29; reviewer
    # item M2 of the Q1 paper). Both need the regular (honest) models; like
    # the original attacks, the attackers are assumed to know them - here the
    # honest nodes' current models, i.e. what each last published.
    # ------------------------------------------------------------------
    def _honest_models(self) -> List[Dict[str, torch.Tensor]]:
        sim = getattr(self.node, "simulator", None)
        nodes = getattr(sim, "nodes", {}) or {}
        return [n.weights for n in nodes.values()
                if getattr(n, "node_type", None) == "honest" and getattr(n, "weights", None)]

    def _attacker_count(self) -> Tuple[int, int]:
        nodes = getattr(getattr(self.node, "simulator", None), "nodes", {}) or {}
        n = len(nodes)
        b = sum(1 for x in nodes.values() if getattr(x, "node_type", None) == "poisoning")
        return n, b

    def _alie_z(self, atk_cfg: Dict[str, Any]) -> float:
        # Baruch et al. (NeurIPS 2019): s = floor(n/2 + 1) - b supporters are
        # needed, z = Phi^-1((n - s) / n). n = 10, b = 2 gives z = 0.253.
        if atk_cfg.get("alie_z") is not None:
            return float(atk_cfg["alie_z"])
        n, b = self._attacker_count()
        s = math.floor(n / 2 + 1) - b
        p = (n - s) / n if n else 0.5
        return NormalDist().inv_cdf(min(max(p, 1e-6), 1 - 1e-6))

    def _stack_stats(self, models: List[Dict[str, torch.Tensor]], key: str):
        t = torch.stack([m[key].float() for m in models])
        return t.mean(dim=0), (t.std(dim=0) if len(models) > 1 else torch.zeros_like(t[0]))

    def _in_band_weights(self, init_weights: Dict[str, Any], mode: str,
                         atk_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        honest = self._honest_models()
        if not honest:
            return None
        out: Dict[str, Any] = {}
        if mode == "alie":
            # ClippedGossip (He et al., App. B.1) decentralized form: publish
            # mu - z * sigma of the regular models, coordinate-wise.
            z = self._alie_z(atk_cfg)
            for k, v in init_weights.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and all(k in m for m in honest):
                    mu, sd = self._stack_stats(honest, k)
                    out[k] = (mu - z * sd).to(v.device).to(v.dtype)
                else:
                    out[k] = copy.deepcopy(v)
            self.node.logger.info(f"[{self.node.node_id}] ALIE: z={z:.3f} over {len(honest)} honest models.")
            return out
        # "ipm": Xie et al. (UAI 2019) send -eps * mean honest update. With models
        # instead of gradients: the honest update since this attacker's previous
        # round is mu - mu_prev; publish the previous mean moved against it,
        # mu_prev - eps * (mu - mu_prev). First round: publish mu.
        eps = float(atk_cfg.get("ipm_eps", 0.1))
        prev = getattr(self, "_ipm_prev_mean", None)
        cur: Dict[str, torch.Tensor] = {}
        for k, v in init_weights.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point() and all(k in m for m in honest):
                mu, _ = self._stack_stats(honest, k)
                cur[k] = mu
                base = prev[k] if prev is not None and k in prev else mu
                out[k] = (base - eps * (mu - base)).to(v.device).to(v.dtype)
            else:
                out[k] = copy.deepcopy(v)
        self._ipm_prev_mean = {k: t.clone() for k, t in cur.items()}
        self.node.logger.info(f"[{self.node.node_id}] IPM: eps={eps} over {len(honest)} honest models.")
        return out

    def train_and_evaluate(self, init_weights: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        self.node.logger.info(f"[{self.node.node_id}] Initiating POISONING attack with Metadata Spoofing.")
        
        poisoned_weights = {}

        # Attack magnitude (2026-09-27, thesis-hub/docs/decisions.md): the noise
        # scale is configurable so a sweep can map where the distance gate stops
        # seeing the attack, down into the in-band regime a stealthy attacker
        # would target. Default 10.0 = the original, hard-coded behaviour.
        atk_cfg = self.node.config.get("participants", {}).get("attacks", {}).get("poisoning", {}) or {}
        noise_std = float(atk_cfg.get("noise_std", 10.0))
        evaluate_true = bool(atk_cfg.get("evaluate_true_accuracy", False))
        # mode: "gaussian" (default, the original attack), "alie" or "ipm".
        mode = str(atk_cfg.get("mode", "gaussian")).lower()

        # Step 1: Inject catastrophic noise into the model weights
        in_band = self._in_band_weights(init_weights, mode, atk_cfg) if mode in ("alie", "ipm") else None
        if in_band is not None:
            poisoned_weights = in_band
        elif mode in ("alie", "ipm"):
            # No honest model known yet (start of the run): publish the received model.
            poisoned_weights = {k: copy.deepcopy(v) for k, v in init_weights.items()}
        else:
            for k, v in init_weights.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    noise = torch.randn_like(v) * noise_std
                    poisoned_weights[k] = v + noise.to(v.device)
                else:
                    poisoned_weights[k] = copy.deepcopy(v)
                
        # Step 2: Simulate physical compute time to evade timing analysis
        base_time = self.node.config.get("simulation", {}).get("base_training_time_per_epoch", 1.0)
        simulated_compute_time = (base_time * self.node.epochs) / self.node.capacity
        
        # Step 3: Ground Truth (For the DES Oracle). At the default scale the
        # model is destroyed and 0.0 is kept; a weaker attack may leave it usable,
        # so the sweep can ask for the real value instead.
        true_accuracy = 0.0
        if evaluate_true:
            model = self.node.cache_manager.checkout_model(poisoned_weights)
            true_accuracy, _ = self.evaluate_state(model)
            self.node.cache_manager.checkin_model(model)
        
        # Step 4: Deception - Spoof metadata to claim high accuracy
        self.spoof_metadata(target_accuracy=96.5)
        
        return poisoned_weights, simulated_compute_time, true_accuracy

class PoisoningNode(BaseNode):
    """
    A standard network participant injected with the Byzantine Poisoning behavior.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = PoisoningTrainer(self)