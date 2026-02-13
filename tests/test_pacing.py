import pytest
from src.bidding.pacing import PacingController
from src.bidding.config import config

@pytest.fixture
def pacing_ctrl():
    return PacingController()

def test_initial_state(pacing_ctrl):
    stats = pacing_ctrl.get_stats()
    assert stats["requests_seen"] == 0
    assert stats["estimated_spend"] == 0.0

def test_pacing_steady(pacing_ctrl):
    """Test steady state (on track)."""
    # Simulate perfect pacing
    # 10% of requests seen, 10% of budget spent
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._estimated_spend = config.pacing.total_budget * 0.10
    
    factor, _ = pacing_ctrl.update(100)
    assert factor == config.pacing.factor_steady

def test_pacing_overspend(pacing_ctrl):
    """Test cool down trigger."""
    # 10% seen, but 20% budget spent (OVERSPEND)
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._estimated_spend = config.pacing.total_budget * 0.20
    
    factor, _ = pacing_ctrl.update(100)
    assert factor == config.pacing.factor_cool_down

def test_pacing_underspend(pacing_ctrl):
    """Test speed up trigger."""
    # 10% seen, but only 1% budget spent (UNDERSPEND)
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._estimated_spend = config.pacing.total_budget * 0.01
    
    factor, _ = pacing_ctrl.update(100)
    assert factor == config.pacing.factor_speed_up
