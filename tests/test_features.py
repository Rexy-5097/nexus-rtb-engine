import pytest
import logging
from src.bidding.features import FeatureExtractor
from src.bidding.schema import BidRequest
from src.bidding.config import config

@pytest.fixture
def mock_bid_request():
    return BidRequest(
        bidId="test_1",
        timestamp="2023",
        visitorId="v1",
        userAgent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        ipAddress="127.0.0.1",
        region="15",
        city="8",
        adExchange="2",
        domain="google.com",
        url="http://google.com",
        anonymousURLID="",
        adSlotID="slot1",
        adSlotWidth="300",
        adSlotHeight="250",
        adSlotVisibility="FirstView",
        adSlotFormat="Fixed",
        adSlotFloorPrice="50",
        creativeID="c1",
        advertiserId="1458",
        userTags="tag1"
    )

def test_feature_extraction_consistency(mock_bid_request):
    """Ensure feature extraction is deterministic."""
    extractor = FeatureExtractor()
    f1 = extractor.extract(mock_bid_request)
    f2 = extractor.extract(mock_bid_request)
    assert f1 == f2
    assert len(f1) == 8  # We define 8 features

def test_user_agent_parsing():
    """Ensure UA parsing handles edge cases."""
    # Mac / Chrome
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    # Testing static method
    os_t, br_t = FeatureExtractor._parse_ua(ua)
    assert os_t == "mac"
    assert br_t == "chrome"

    # Unknown
    os_t, br_t = FeatureExtractor._parse_ua(None)
    assert os_t == "unknown"
    assert br_t == "unknown"

def test_hash_bounds(mock_bid_request):
    """Ensure hashes stay within HASH_SPACE."""
    extractor = FeatureExtractor()
    features = extractor.extract(mock_bid_request)
    for f in features:
        assert 0 <= f < config.model.hash_space
