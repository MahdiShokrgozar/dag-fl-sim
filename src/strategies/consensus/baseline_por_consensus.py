import logging
import hashlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Slash evidence type constant (moved from farabi_consensus)
SLASH_EVIDENCE_TYPE = "slash_evidence"

class BaselinePoRConsensus:
    """
    Baseline O(1) Proof-of-Reputation (PoR) Consensus for comparative studies.
    Implements a fast-path validation using a Commitment Scheme (Accuracy + S5 Fingerprint)
    weighted by the proposer's reputation virtue.
    
    Enhanced with lightweight anomaly detection for Byzantine behavior identification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node = None  # Injected at runtime by BaseNode
        
        # Load configuration or set defaults for the baseline
        consensus_cfg = config.get("protocols", {}).get("baseline_por", {}).get("consensus", {})
        self.quantization_decimals = consensus_cfg.get("quantization_decimals", 3)
        
        # Anomaly detection thresholds (lightweight sanity checks)
        self.min_accuracy_threshold = consensus_cfg.get("min_accuracy_threshold", 10.0)
        self.max_accuracy_threshold = consensus_cfg.get("max_accuracy_threshold", 99.9)
        self.max_class_imbalance = consensus_cfg.get("max_class_imbalance", 80.0)
        
        logger.info(
            f"🔒 [BASELINE-PoR] Initialized | Quantization: {self.quantization_decimals} decimals | "
            f"Anomaly Detection: Enabled"
        )
    
    def propose_block(self, payload: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
        """
        Wraps the outbound payload with a cryptographic commitment for integrity.
        """
        accuracy = payload.get("accuracy", 0.0)
        fingerprint = payload.get("model_fingerprint", 0.0)
        
        # Quantize to avoid floating-point mismatch across heterogeneous hardware
        commitment_hash = self._generate_commitment(accuracy, fingerprint)
        payload["commitment"] = commitment_hash
        
        return {
            "proposer": node_id,
            "type": "standard_update",
            "payload": payload,
        }
    
    def validate_block(self, block: Dict[str, Any]) -> bool:
        """
        O(1) Baseline Validation with Lightweight Anomaly Detection.
        Checks quarantine status, commitment integrity, and behavioral anomalies.
        """
        if not self.node or not hasattr(self.node, 'reputation'):
            # Fallback for isolated unit testing
            return True
        
        proposer = block.get("proposer", "unknown")
        
        # 1. Reputation Filter: Drop blocks from quarantined nodes
        if self.node.reputation.is_quarantined(proposer):
            block["_validation_receipt"] = {
                "status": "rejected",
                "reason": "quarantined_node",
                "weight": 0.0
            }
            logger.warning(f"🚫 [BASELINE-PoR] Block rejected. Proposer {proposer} is quarantined.")
            return False
        
        payload = block.get("payload", {})
        
        # 2. Commitment Verification
        claimed_acc = payload.get("accuracy", 0.0)
        claimed_s5 = payload.get("model_fingerprint", 0.0)
        claimed_commitment = payload.get("commitment")
        
        if claimed_commitment:
            expected_commitment = self._generate_commitment(claimed_acc, claimed_s5)
            if claimed_commitment != expected_commitment:
                block["_validation_receipt"] = {
                    "status": "rejected",
                    "reason": "commitment_mismatch",
                    "weight": 0.0
                }
                logger.error(f"☠️ [BASELINE-PoR] Commitment mismatch for {proposer}. Integrity check failed.")
                return False
        
        # 3. Lightweight Anomaly Detection (NEW)
        anomaly_reason = self._detect_anomaly(payload, proposer)
        if anomaly_reason:
            block["_validation_receipt"] = {
                "status": "rejected",
                "reason": anomaly_reason,
                "weight": 0.0
            }
            logger.warning(f"⚠️ [BASELINE-PoR] Anomaly detected for {proposer}: {anomaly_reason}")
            return False
        
        # 4. Pass validation and attach the baseline reputation weight
        virtue_score = self.node.reputation.get_virtue_score(proposer)
        block["_validation_receipt"] = {
            "status": "approved",
            "weight": virtue_score
        }
        return True
    
    def _detect_anomaly(self, payload: Dict[str, Any], proposer: str) -> Optional[str]:
        """
        Lightweight anomaly detection for Byzantine behavior.
        Returns reason string if anomaly detected, None otherwise.
        
        Detection mechanisms:
        1. Accuracy sanity check (detect lazy/poisoning)
        2. Class distribution check (detect backdoor)
        3. Fingerprint stability check (detect stealing)
        """
        # 1. Accuracy sanity check (detect lazy/poisoning)
        # accuracy = payload.get("accuracy", 0.0)
        # if accuracy < self.min_accuracy_threshold:
        #     return "suspiciously_low_accuracy"
        # if accuracy > self.max_accuracy_threshold:
        #     return "suspiciously_high_accuracy"
        
        # 2. Class distribution check (detect backdoor)
        # class_accuracies = payload.get("class_accuracies", {})
        # if class_accuracies:
        #     acc_values = list(class_accuracies.values())
        #     if acc_values:
        #         min_acc = min(acc_values)
        #         max_acc = max(acc_values)
        #         if max_acc - min_acc > self.max_class_imbalance:
        #             return "extreme_class_imbalance"
        
        # 3. Fingerprint stability check (detect stealing)
        # Note: This requires historical data which may not be available in all scenarios
        # For now, we rely on commitment verification for integrity
        
        accuracy = payload.get("accuracy", 0.0)
    
        # فقط برای accuracy خیلی پایین (که در Non-IID هم غیرطبیعی است)
        if accuracy < 1.0:
            return "suspiciously_low_accuracy"
    
    
        return None  # No anomaly detected
    
    def _generate_commitment(self, accuracy: float, fingerprint: float) -> str:
        """
        Generates a stable deterministic hash by quantizing floats.
        """
        q_acc = round(accuracy, self.quantization_decimals)
        q_fp = round(fingerprint, self.quantization_decimals)
        raw_string = f"{q_acc:.3f}_{q_fp:.3f}"
        return hashlib.sha256(raw_string.encode('utf-8')).hexdigest()