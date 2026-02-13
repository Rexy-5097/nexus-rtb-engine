import logging
import hashlib
from typing import List, Tuple
from src.bidding.schema import BidRequest
from src.bidding.config import config
from src.utils.hashing import hash_feature

# Configure logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Handles feature extraction and hashing for the bidding model.
    Encapsulates logic for parsing User-Agent, normalization, and token generation.
    """

    @staticmethod
    def _norm(val: object) -> str:
        """Normalize input strings to handle 'NaN', 'null', empty strings."""
        if not val:
            return "unknown"
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return "unknown"
        return s

    @staticmethod
    def _parse_ua(ua: str) -> Tuple[str, str]:
        """Parse User-Agent string into OS and Browser categories."""
        if not ua or ua == "unknown":
            return "unknown", "unknown"
        
        ua_lower = ua.lower()
        
        # OS Detection
        if "windows" in ua_lower: os_t = "windows"
        elif "mac" in ua_lower: os_t = "mac"
        elif "ios" in ua_lower: os_t = "ios"
        elif "android" in ua_lower: os_t = "android"
        elif "linux" in ua_lower: os_t = "linux"
        else: os_t = "other"
        
        # Browser Detection
        if "edge" in ua_lower: br_t = "edge"
        elif "chrome" in ua_lower: br_t = "chrome"
        elif "firefox" in ua_lower: br_t = "firefox"
        elif "safari" in ua_lower: br_t = "safari"
        elif "msie" in ua_lower or "trident" in ua_lower: br_t = "ie"
        elif "opera" in ua_lower: br_t = "opera"
        else: br_t = "other"
        
        return os_t, br_t

    def extract(self, request: BidRequest) -> List[int]:
        """
        Extract features from a BidRequest and return a list of hashed feature indices.
        
        Args:
            request (BidRequest): The incoming request object.
            
        Returns:
            List[int]: Sparse vector of feature indices.
        """
        # Normalize inputs
        adv_id = str(request.advertiserId)
        ua = self._norm(request.userAgent)
        region = self._norm(request.region)
        city = self._norm(request.city)
        domain = self._norm(request.domain)
        vis = self._norm(request.adSlotVisibility)
        fmt = self._norm(request.adSlotFormat)
        
        os_t, br_t = self._parse_ua(ua)
        
        # Consistent feature string format
        # Note: These keys must match training pipeline exactly
        raw_features = [
            f"ua_os:{os_t}",
            f"ua_browser:{br_t}",
            f"region:{region}",
            f"city:{city}",
            f"adslot_visibility:{vis}",
            f"adslot_format:{fmt}",
            f"advertiser:{adv_id}",
            f"domain:{domain}",
        ]
        
        # Hash features
        # Using 2^18 = 262144 bucket space
        hash_space = config.model.hash_space
        hashed_features = [
            hash_feature(f, hash_space) 
            for f in raw_features
        ]
        
        return hashed_features
