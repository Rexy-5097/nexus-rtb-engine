import pytest
from src.bidding.pacing import PacingController
from src.bidding.config import config

@pytest.fixture
def pacing_ctrl():
    return PacingController()

def test_initial_state(pacing_ctrl):
    stats = pacing_ctrl.get_stats()
    # requests counter might be renamed or we check internal
    # We renamed key to "requests" in get_stats
    assert stats["requests"] == 0.0

def test_pacing_steady(pacing_ctrl):
    """Test steady state (on track)."""
    # Simulate perfect pacing
    # 10% of requests seen, 10% of budget spent
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._spent_budget = config.pacing.total_budget * 0.10
    
    factor = pacing_ctrl.update(100) # 100 is intended spend
    # Should be close to initial alpha or stable
    assert 0.1 <= factor <= 2.0

def test_pacing_overspend(pacing_ctrl):
    """Test cool down trigger."""
    # 10% seen, but 20% budget spent (OVERSPEND)
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._spent_budget = config.pacing.total_budget * 0.20
    
    factor = pacing_ctrl.update(100)
    # Should decrease alpha
    assert factor < config.alpha_initial

def test_pacing_underspend(pacing_ctrl):
    """Test speed up trigger."""
    # 10% seen, but only 1% budget spent (UNDERSPEND)
    pacing_ctrl._requests_seen = int(config.pacing.expected_requests * 0.10)
    pacing_ctrl._spent_budget = config.pacing.total_budget * 0.01
    
    factor = pacing_ctrl.update(100)
    # Should increase alpha
    assert factor > config.alpha_initial
