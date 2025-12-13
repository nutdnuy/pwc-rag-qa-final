from pathlib import Path
from qa import answer_question

def test_vpn():
    res = answer_question(Path("policy.txt"), "Is VPN required for all external connections?", refuse_threshold=0.0)
    assert "VPN" in res["answer"]
