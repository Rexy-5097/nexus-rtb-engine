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
    # Create real small arrays
    import numpy as np
    data = {
        "ctr": {"coef": np.zeros((1, config.model.hash_space)), "intercept": [-1.0]},
        "cvr": {"coef": np.zeros((1, config.model.hash_space)), "intercept": [-1.0]},
        "stats": {"1458": {"avg_mp": 50.0, "avg_ev": 0.5}},
        "scaler": {},
        "calibration": {"a": 1.0, "b": 0.0},
        "adv_priors": {"1458": 0.001},
        "n_map": {"1458": 10},
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
        
    # Patch signature verification to always pass for this file
    with patch("src.utils.crypto.ModelIntegrity.verify_signature", return_value=True):
        yield str(model_path)

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
    # Ensure feature extractor returns tuples
    # Actually engine uses real extractor unless mocked. Real one now returns tuples.
    response = engine.process(valid_request)
    if response.explanation and "pacing" in response.explanation:
        # Pacing might block it if we are extremely unlucky with randoms?
        # But mock model returns low prob/EV, so bid might be small.
        pass
    # We just want to ensure it runs without error
    assert response.bidId == "bid_1"
    assert response.advertiserId == "1458"
    assert isinstance(response.bidPrice, int)

def test_floor_price_rejection(engine, valid_request):
    """Ensure bids below floor are rejected."""
    # relaxed_config = replace(config, max_cpa=1000.0) 
    # Logic changed. Now we use EV < floor check.
    # To pass guards, we need high EV.
    # pCTR ~ sigmoid(-2) ~ 0.12. N=10. pCVR~0.12.
    # EV ~ 0.12 * 50 + (0.12*0.12) * 500 = 6 + 7.2 = 13.2.
    # bid ~ 13 * alpha.
    
    with patch("src.bidding.engine.config", config): # Use default
         # Mock params to get high EV
         with patch.object(engine.model_loader, 'intercept_ctr', 2.0): # p~0.88
             with patch.object(engine.model_loader, 'intercept_cvr', 2.0):
                with patch.object(engine.model_loader, 'model_loaded', True):
                    # Mock stats to pass stability guard
                    with patch.object(engine.model_loader, 'get_stats', return_value={"avg_mp": 1.0, "avg_ev": 1.0}):
                        # EV will be high (~ 400).
                        # Floor = 5000.
                        valid_request.adSlotFloorPrice = "5000"
                        response = engine.process(valid_request)
                        assert response.bidPrice == 0
                        assert response.explanation == "below_floor"

# Test removed: test_roi_guard (superseded by new logic checks)
# Logic: roi_safety_violation (custom vs ev), stability_guard (ev vs mp), quality_gate (ev vs avg_ev).

def test_quality_gate(engine, valid_request):
    """Ensure low EV bids are rejected."""
    # Mock model to return super low probability (-20 => ~0)
    with patch.object(engine.model_loader, 'intercept_ctr', -20.0):
        with patch.object(engine.model_loader, 'intercept_cvr', -20.0):
             # Mock avg_ev to be high so check fails
             with patch.object(engine.model_loader, 'get_stats', return_value={"avg_ev": 100.0, "avg_mp": 50.0}):
                response = engine.process(valid_request)
                assert response.bidPrice == 0
                assert response.explanation == "quality_gate"


def test_fail_closed_model(engine, valid_request):
    """Ensure engine returns 0 if model is not loaded."""
    # Force model_loaded = False
    with patch.object(engine.model_loader, 'model_loaded', False):
        response = engine.process(valid_request)
        assert response.bidPrice == 0
        assert response.explanation == "model_not_loaded"
