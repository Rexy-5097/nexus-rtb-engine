from typing import List, Tuple
from src.bidding.schema import BidRequest
from src.bidding.config import config
from src.utils.hashing import Hasher

class FeatureExtractor:
    """
    Handles feature extraction and hashing for the bidding model.
    Encapsulates logic for parsing User-Agent, normalization, and token generation.
    """

    @staticmethod
    def _norm(val: str) -> str:
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

    @classmethod
    def extract(cls, request: BidRequest) -> List[int]:
        """
        Extract features from a BidRequest and return a list of hashed feature indices.
        """
        adv_id = str(request.advertiserId)
        ua = cls._norm(request.userAgent)
        region = cls._norm(request.region)
        city = cls._norm(request.city)
        domain = cls._norm(request.domain)
        vis = cls._norm(request.adSlotVisibility)
        fmt = cls._norm(request.adSlotFormat)
        
        os_t, br_t = cls._parse_ua(ua)
        
        # Consistent feature string format
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
        hashed_features = [
            Hasher.adler32_hash(f, config.model.hash_space) 
            for f in raw_features
        ]
        
        return hashed_features
