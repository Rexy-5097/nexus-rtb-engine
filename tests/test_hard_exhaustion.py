import pytest
from unittest.mock import MagicMock, patch
from src.bidding.engine import BiddingEngine
from src.bidding.pacing import PacingController
from src.bidding.schema import BidRequest
from src.bidding.config import config
from dataclasses import replace

@pytest.fixture
def mock_model_file(tmp_path):
    import pickle
    import numpy as np
    model_path = tmp_path / "model_weights.pkl"
    data = {
        "ctr": {"coef": np.zeros((1, 10)), "intercept": [0.0]},
        "cvr": {"coef": np.zeros((1, 10)), "intercept": [0.0]},
        "stats": {"123": {"avg_mp": 50.0, "avg_ev": 0.5}}
    }
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
    return str(model_path)

@pytest.fixture
def engine(mock_model_file):
    return BiddingEngine(model_path=mock_model_file)

@pytest.fixture
def request_stub():
    return BidRequest(
        bidId="bid_1", timestamp="123", visitorId="v1", userAgent="Mozilla",
        ipAddress="1.1.1.1", region="1", city="1", adExchange="1",
        domain="d1", url="u1", anonymousURLID="a1",
        adSlotID="1", adSlotWidth="300", adSlotHeight="250",
        adSlotVisibility="1", adSlotFormat="1", adSlotFloorPrice="10",
        creativeID="cr1", advertiserId="123", userTags="t1"
    )

def test_hard_exhaustion(engine, request_stub):
    """
    Test Step 4:
    1. Initialize budget = 100
    2. Reserve 100
    3. Next reserve attempt must return False
    4. Engine get_bid() must return 0
    5. remaining_budget must equal 0
    """
    # 1. Initialize budget = 100
    engine.pacing._total_budget = 100.0
    engine.pacing._spent_budget = 0.0
    
    # 2. Reserve 100
    # Reserve exactly the remaining amount
    success = engine.pacing.reserve_budget(100.0)
    assert success is True
    assert engine.pacing.is_exhausted() is True
    assert engine.pacing.remaining_budget == 0.0
    
    # 3. Next reserve attempt must return False
    success_2 = engine.pacing.reserve_budget(1.0)
    assert success_2 is False
    
    # 4. Engine process() must return 0
    # Ideally should hit "budget_exhausted" check at top
    response = engine.process(request_stub)
    assert response.bidPrice == 0
    assert response.explanation == "budget_exhausted"
    
    # 5. remaining_budget must equal 0
    assert engine.pacing.remaining_budget == 0.0

def test_soft_cap_transparency(engine):
    """
    Verify soft caps do NOT block reservation (per new logic).
    """
    engine.pacing._total_budget = 1000.0
    engine.pacing._spent_budget = 0.0
    
    # Set hourly cap low
    engine.pacing.conf = replace(engine.pacing.conf, max_hourly_spend=10.0)
    
    # We exceed hourly cap
    engine.pacing._hourly_spend = 15.0
    
    # But reserve_budget should still succeed if global budget exists
    success = engine.pacing.reserve_budget(5.0)
    assert success is True
