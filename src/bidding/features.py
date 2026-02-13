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

    def __init__(self):
        self.top_k_maps = {} # {feature: set(values)}

    def set_encoding_maps(self, maps: dict):
        """Load Top-K maps for hybrid encoding."""
        self.top_k_maps = {k: set(v) for k,v in maps.items()}

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

    @staticmethod
    def _entropy(s: str) -> float:
        """Calculate Shannon entropy of string."""
        if not s: return 0.0
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        import math
        return -sum(p * math.log(p) / math.log(2.0) for p in prob)

    def extract(self, request: BidRequest, scaler: dict = None, stats: dict = None) -> List[Tuple[int, float]]:
        """
        Extract features from a BidRequest and return a list of (hash, value) tuples.
        """
        # Normalize inputs
        adv_id = str(request.advertiserId)
        ua = self._norm(request.userAgent)
        region = self._norm(request.region)
        city = self._norm(request.city)
        domain = self._norm(request.domain)
        vis = self._norm(request.adSlotVisibility)
        fmt = self._norm(request.adSlotFormat)
        
        # 1. Ad Slot Area & Floor Bucket
        def safe_float(v):
            try: return float(v)
            except: return 0.0

        width = safe_float(request.adSlotWidth)
        height = safe_float(request.adSlotHeight)
        area = width * height
        
        floor = safe_float(request.adSlotFloorPrice)
        if floor <= 0: floor_bucket = "0"
        elif floor <= 10: floor_bucket = "1-10"
        elif floor <= 50: floor_bucket = "10-50"
        elif floor <= 100: floor_bucket = "50-100"
        else: floor_bucket = "100+"
        
        os_t, br_t = self._parse_ua(ua)
        ua_entropy = self._entropy(ua)
        
        # 2. Time Features (Cyclic)
        try:
            ts = int(float(request.timestamp))
            if ts > 1000000000000: ts //= 1000
            import datetime
            import math
            dt = datetime.datetime.fromtimestamp(ts)
            hour = dt.hour
            weekday = dt.weekday()
            
            # Cyclic Encoding
            hour_sin = math.sin(2 * math.pi * hour / 24.0)
            hour_cos = math.cos(2 * math.pi * hour / 24.0)
        except:
            hour, weekday = 0, 0
            hour_sin, hour_cos = 0.0, 1.0
            
        hour_bucket = f"hour:{hour}"
        weekday_bucket = f"weekday:{weekday}"

        # 3. New Historical Signals (Rolling & Smoothed)
        # Defaults
        adv_ctr_1d = 0.0
        adv_ctr_7d = 0.0
        dom_ctr_1d = 0.0
        dom_ctr_7d = 0.0
        adv_win_rate = 0.0
        adv_avg_cpm = 0.0
        slot_ctr = 0.0
        
        stat_adv_ctr = 0.0
        stat_adv_dom_ctr = 0.0
        dom_freq = 0.0
        
        if stats:
            # Helper for Bayesian Smoothing
            def smooth(clicks, imps, alpha=10, beta=400):
                return (clicks + alpha) / (imps + beta)

            # Global Stats (Fallback)
            ak = str(adv_id) 
            if ak in stats:
                s = stats[ak] # [imps, spend, clicks, convs]
                if s[0] > 0: stat_adv_ctr = smooth(s[2], s[0])
            
            dk = f"domain:{domain}"
            if dk in stats:
                d_s = stats[dk]
                dom_freq = float(d_s[0])
                if d_s[0] > 0: stat_adv_dom_ctr = smooth(d_s[2], d_s[0])
            
            # --- ROLLING STATS (Lookups) ---
            # Keys expected in stats dict (or request attributes for training)
            
            def get_stat(key, attr_name):
                # Training: Check request attribute first (Point-in-Time)
                val = getattr(request, attr_name, None)
                if val is not None: return float(val)
                # Inference: Check stats dict (Snapshot)
                return float(stats.get(key, 0.0))
            
            # Example: stats["adv_1d:1458"] = CTR_Value
            adv_ctr_1d = get_stat(f"adv_1d:{adv_id}", "adv_ctr_1d")
            adv_ctr_7d = get_stat(f"adv_7d:{adv_id}", "adv_ctr_7d")
            dom_ctr_1d = get_stat(f"dom_1d:{domain}", "dom_ctr_1d")
            dom_ctr_7d = get_stat(f"dom_7d:{domain}", "dom_ctr_7d")
            
            adv_win_rate = get_stat(f"adv_win:{adv_id}", "adv_win_rate") 
            adv_avg_cpm = get_stat(f"adv_cpm:{adv_id}", "adv_avg_cpm")
            slot_ctr = get_stat(f"slot:{fmt}_{vis}", "slot_ctr") 
            
            # Phase 7: User History Signals (if stats provided and contains user)
            # Only if user_ctr is available in request or stats
            user_ctr = get_stat(f"user:{request.visitorId}", "user_ctr")
            user_count = get_stat(f"user_cnt:{request.visitorId}", "user_count_7d")

        else:
            user_ctr = 0.0
            user_count = 0.0

        # Helper: Hybrid Encoding (Top-K vs Tail)
        def encode_cat(name, val):
            if name in self.top_k_maps:
                if val in self.top_k_maps[name]:
                    return f"{name}:{val}"
                else:
                    return f"{name}:tail" # Collapse tail
            return f"{name}:{val}" 

        # Cross Features
        cross_features = [
            f"cross_region_adv:{region}_{adv_id}",
            f"cross_city_adv:{city}_{adv_id}",
            f"cross_os_adv:{os_t}_{adv_id}",
            f"cross_browser_adv:{br_t}_{adv_id}",
            f"cross_domain_adv:{domain}_{adv_id}",
            f"cross_floor_adv:{floor_bucket}_{adv_id}",
        ]
        
        # Feature Lists
        # Numeric Features (Scaled)
        numeric_features = {
            "region": safe_float(region),
            "city": safe_float(city),
            "adslot_visibility": safe_float(vis),
            "adslot_format": safe_float(fmt),
            "ad_slot_area": area,
            
            # Global (Smoothed)
            "stat_adv_ctr": stat_adv_ctr,
            "stat_adv_dom_ctr": stat_adv_dom_ctr,
            
            # Rolling (Time-Aware)
            "adv_ctr_1d": adv_ctr_1d,
            "adv_ctr_7d": adv_ctr_7d,
            "dom_ctr_1d": dom_ctr_1d,
            "dom_ctr_7d": dom_ctr_7d,
            "adv_win_rate": adv_win_rate,
            "adv_avg_cpm": adv_avg_cpm,
            "slot_ctr": slot_ctr,
            
            "ua_entropy": ua_entropy,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            
            # Phase 7
            "domain_freq": dom_freq,
            "user_ctr": user_ctr,
            "user_click_count_7d": user_count,
        }
        
        # Categorical Features (One-Hot / Hybrid)
        categorical_features = [
            encode_cat("ua_os", os_t),
            encode_cat("ua_browser", br_t),
            encode_cat("advertiser", adv_id),
            encode_cat("domain", domain),
            encode_cat("region", region),
            encode_cat("city", city),
            f"floor_bucket:{floor_bucket}",
            hour_bucket,
            weekday_bucket,
        ] + cross_features
        
        # Hashing
        hash_space = config.model.hash_space
        hashed_features = []
        
        for f in categorical_features:
            h = hash_feature(f, hash_space)
            hashed_features.append((h, 1.0))
            
        for name, val in numeric_features.items():
            h = hash_feature(name, hash_space)
            # Use distinct hash space region for numeric? (Optional, but usually okay to mix if sparse)
            # Scale numeric
            scaled_val = val
            if scaler and name in scaler:
                mean, std = scaler[name]
                if std > 0:
                    scaled_val = (val - mean) / std
            hashed_features.append((h, scaled_val))
            
        return hashed_features
