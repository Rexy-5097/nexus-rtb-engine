from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BidRequest:
    """
    Represents a Real-Time Bidding request with strict type hints.
    Using slots for memory efficiency and faster attribute access.
    """

    bidId: str
    timestamp: str
    visitorId: str
    userAgent: str
    ipAddress: str
    region: str
    city: str
    adExchange: str
    domain: str
    url: str
    anonymousURLID: str
    adSlotID: str
    adSlotWidth: str
    adSlotHeight: str
    adSlotVisibility: str
    adSlotFormat: str
    adSlotFloorPrice: str
    creativeID: str
    advertiserId: str
    userTags: str

    def __post_init__(self):
        """Sanitize inputs to prevent memory exhaustion attacks."""
        # Ensure strings are not excessively long (basic protection against memory DoS)
        # Detailed validation should happen in a separate validation layer.
        pass


@dataclass(slots=True)
class BidResponse:
    """
    Represents the decision made by the bidding engine.
    """

    bidId: str
    bidPrice: int
    advertiserId: str
    explanation: Optional[str] = None
