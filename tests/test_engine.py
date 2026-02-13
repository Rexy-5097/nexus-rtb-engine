import pytest
import os
import pickle
from unittest.mock import MagicMock, patch

from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest
from src.bidding.config import config

@pytest.fixture
def mock_model_file(tmp_path):
    """Create a dummy model file for testing."""
    model_path = tmp_path / "model_weights.pkl"
    data = {
        "ctr": {"coef": MagicMock(flatten=lambda: None), "intercept": [-1.0]},
        "cvr": {"coef": None, "intercept": [-1.0]},
        "stats": {"1458": {"avg_mp": 50.0, "avg_ev": 0.5}}
    }
    # Mocking coef as None to test intercept-only logic or mocked arrays
    # Actually let's create real (small) numpy arrays for validity
    import numpy as np
    data["ctr"]["coef"] = np.zeros((18, config.model.hash_space))
    data["cvr"]["coef"] = np.zeros((18, config.model.hash_space))
    
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
    return str(model_path)

@pytest.fixture
def engine(mock_model_file):
    return BiddingEngine(model_path=mock_model_file)

@pytest.fixture
def valid_request():
    return BidRequest(
        bidId="bid_1", timestamp="123", visitorId="v1", userAgent="Mozilla",
        ipAddress="1.1.1.1", region="1", city="1", adExchange="1",
        domain="clean.com", url="http://clean.com", anonymousURLID="",
        adSlotID="1", adSlotWidth="300", adSlotHeight="250",
        adSlotVisibility="1", adSlotFormat="1", adSlotFloorPrice="10",
        creativeID="cr1", advertiserId="1458", userTags="t1"
    )

def test_engine_bid_flow(engine, valid_request):
    """Test full bid flow returns a response."""
    response = engine.process(valid_request)
    assert response.bidId == "bid_1"
    assert response.advertiserId == "1458"
    assert isinstance(response.bidPrice, int)

def test_floor_price_rejection(engine, valid_request):
    """Ensure bids below floor are rejected."""
    # Set floor very high
    valid_request.adSlotFloorPrice = "5000"
    response = engine.process(valid_request)
    assert response.bidPrice == -1
    assert response.explanation == "below_floor"

def test_quality_gate(engine, valid_request):
    """Ensure low EV bids are rejected."""
    # Mock model to return super low probability
    with patch.object(engine.model_loader, 'intercept_ctr', -20.0): # ~0 probability
        with patch.object(engine.model_loader, 'intercept_cvr', -20.0):
            response = engine.process(valid_request)
            assert response.bidPrice == -1
            assert response.explanation == "quality_gate"
