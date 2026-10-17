import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

# فرض می‌کنیم کلاس در این مسیر ذخیره شده است
from src.strategies.consensus.baseline_por_consensus import BaselinePoRConsensus

@pytest.fixture
def setup_consensus():
    """
    Setup the consensus module and a mock node with a mock reputation system.
    """
    config = {
        "protocols": {
            "baseline_por": {
                "consensus": {
                    "quantization_decimals": 3
                }
            }
        }
    }
    consensus = BaselinePoRConsensus(config)
    
    # Mocking the BaseNode and Reputation
    mock_node = MagicMock()
    mock_reputation = MagicMock()
    
    # Default honest behavior
    mock_reputation.is_quarantined.return_value = False
    mock_reputation.get_virtue_score.return_value = 85.0
    
    mock_node.reputation = mock_reputation
    consensus.node = mock_node
    
    return consensus, mock_node, mock_reputation

def test_propose_block_generates_commitment(setup_consensus):
    """بررسی تولید صحیح بلوک و طرح تعهد"""
    consensus, _, _ = setup_consensus
    
    payload = {
        "accuracy": 95.12345, # دقت بالا
        "model_fingerprint": 12345.6789 # اثر انگشت
    }
    
    block = consensus.propose_block(payload, "node_1")
    
    assert block["proposer"] == "node_1"
    assert "commitment" in block["payload"]
    assert block["payload"]["accuracy"] == 95.12345

def test_validate_block_success(setup_consensus):
    """مسیر استاندارد: گره صادق و داده‌های دست‌نخورده"""
    consensus, _, mock_reputation = setup_consensus
    
    payload = {
        "accuracy": 95.123,
        "model_fingerprint": 12345.678
    }
    block = consensus.propose_block(payload, "node_1")
    
    # اعتبارسنجی باید موفقیت‌آمیز باشد
    is_valid = consensus.validate_block(block)
    
    assert is_valid is True
    assert "_validation_receipt" in block
    assert block["_validation_receipt"]["status"] == "approved"
    assert block["_validation_receipt"]["weight"] == 85.0 # امتیازی که در Fixture تنظیم کردیم

def test_validate_block_rejects_quarantined_node(setup_consensus):
    """دفاع تهاجمی: رد کردن گره قرنطینه‌شده"""
    consensus, _, mock_reputation = setup_consensus
    
    # شبیه‌سازی وضعیت قرنطینه
    mock_reputation.is_quarantined.return_value = True
    
    block = {
        "proposer": "malicious_node",
        "payload": {}
    }
    
    is_valid = consensus.validate_block(block)
    assert is_valid is False

def test_validate_block_rejects_tampered_payload(setup_consensus):
    """دفاع تهاجمی: دستکاری متادیتا بعد از امضا"""
    consensus, _, _ = setup_consensus
    
    payload = {
        "accuracy": 90.000,
        "model_fingerprint": 1000.000
    }
    block = consensus.propose_block(payload, "node_2")
    
    # شبیه‌سازی حمله: تغییر دقت در طول شبکه بدون تغییر تعهد (Commitment)
    block["payload"]["accuracy"] = 99.999 
    
    is_valid = consensus.validate_block(block)
    assert is_valid is False # باید به دلیل عدم تطابق هش رد شود