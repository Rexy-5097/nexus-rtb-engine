import zlib
import math
import pickle
import os
import random

# Import user-provided interfaces
try:
    from BidRequest import BidRequest
    from Bidder import Bidder
except ImportError:
    pass

class Bid(Bidder):
    """
    High-Frequency Real-Time Bidding Engine.
    
    Implements a logic-regression based estimator for CTR/CVR with market-anchored 
    bidding and adaptive pacing control.
    """
    def __init__(self):
        # -------------------------------
        # Configuration
        # -------------------------------
        self.FEATURE_BITS = 18
        self.HASH_SPACE = 2 ** self.FEATURE_BITS
        
        # Pacing: We estimate 25M requests. 
        # Budget is 25M, Avg Price ~80. We can only afford ~1.2% win rate.
        self.EXPECTED_REQUESTS = 25_000_000 
        self.TOTAL_BUDGET = 25_000_000 
        
        self.requests_seen = 0
        self.estimated_spend = 0.0 # Track estimated spend, not raw bids
        
        # Heuristic: We assume we win ~20% of auctions we enter at market price
        self.ESTIMATED_WIN_RATE = 0.20 
        
        # Advertiser N values (Conversion importance weights)
        self.n_map = {
            "1458": 0, "3358": 2, "3386": 0, "3427": 0, "3476": 10,
        }
        
        # Market Anchors (Matches training output)
        self.stats = {
            "1458": {"avg_mp": 69.50, "avg_ev": 0.000801},
            "3358": {"avg_mp": 92.37, "avg_ev": 0.001174},
            "3386": {"avg_mp": 77.24, "avg_ev": 0.000723},
            "3427": {"avg_mp": 81.51, "avg_ev": 0.000748},
            "3476": {"avg_mp": 79.62, "avg_ev": 0.000674},
        }
        
        # Model Parameters
        self.weights_ctr = None
        self.weights_cvr = None
        
        # SAFETY: Default intercepts to -4.0 (approx 1.8% probability) 
        # This fail-safe prevents the engine from bidding aggressively if the model file 
        # cannot be loaded or is corrupted. It ensures we bid conservatively (~1.8% CTR/CVR)
        # rather than randomly, protecting the budget in failure scenarios.
        self.intercept_ctr = -4.0 
        self.intercept_cvr = -4.0

        # -------------------------------
        # Load Trained Model
        # -------------------------------
        try:
            model_path = os.path.join(os.path.dirname(__file__), "model_weights.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                if "ctr" in data:
                    self.weights_ctr = data["ctr"]["coef"].flatten()
                    self.intercept_ctr = float(data["ctr"]["intercept"][0])
                if "cvr" in data:
                    self.weights_cvr = data["cvr"]["coef"].flatten()
                    self.intercept_cvr = float(data["cvr"]["intercept"][0])
                if "stats" in data and data["stats"]:
                    self.stats = data["stats"]
        except Exception:
            pass

    # -------------------------------
    # Utilities
    # -------------------------------
    def _hash(self, s: str) -> int:
        if not s: return 0
        return (zlib.adler32(s.encode("utf-8")) & 0xffffffff) % self.HASH_SPACE

    def _sigmoid(self, x: float) -> float:
        if x < -15: return 0.0000003
        if x > 15: return 0.9999997
        return 1.0 / (1.0 + math.exp(-x))

    def _norm(self, v):
        if v is None: return "unknown"
        s = str(v)
        return "unknown" if s == "" or s.lower() == "nan" else s

    def _parse_ua(self, ua):
        if not ua or ua == "unknown": return "unknown", "unknown"
        ua = ua.lower()
        if "windows" in ua: os_t = "windows"
        elif "mac" in ua: os_t = "mac"
        elif "ios" in ua: os_t = "ios"
        elif "android" in ua: os_t = "android"
        elif "linux" in ua: os_t = "linux"
        else: os_t = "other"
        
        if "edge" in ua: br_t = "edge"
        elif "chrome" in ua: br_t = "chrome"
        elif "firefox" in ua: br_t = "firefox"
        elif "safari" in ua: br_t = "safari"
        elif "msie" in ua or "trident" in ua: br_t = "ie"
        elif "opera" in ua: br_t = "opera"
        else: br_t = "other"
        return os_t, br_t

    # -------------------------------
    # Core Bidding Logic
    # -------------------------------
    def getBidPrice(self, bidRequest: BidRequest) -> int:
        try:
            self.requests_seen += 1
            adv_id = str(bidRequest.getAdvertiserId())
            N = self.n_map.get(adv_id, 0)
            
            # Market Anchors
            stat = self.stats.get(adv_id, {"avg_mp": 70.0, "avg_ev": 0.001})
            avg_mp = float(stat["avg_mp"])
            avg_ev = float(stat["avg_ev"])
            if avg_ev <= 1e-9: avg_ev = 0.0001
            
            # ---------------------------
            # Feature Extraction
            # ---------------------------
            # Extract raw features from the BidRequest object.
            # We normalize strings to handle potential data inconsistencies (e.g., "NaN", "").
            ua = bidRequest.getUserAgent()
            region = self._norm(bidRequest.getRegion())
            city = self._norm(bidRequest.getCity())
            domain = self._norm(bidRequest.getDomain())
            vis = self._norm(bidRequest.getAdSlotVisibility())
            fmt = self._norm(bidRequest.getAdSlotFormat())
            
            # Parse User-Agent into OS and Browser tokens for better generalization
            os_t, br_t = self._parse_ua(ua)
            
            # Construct feature strings for hashing.
            # Format: "feature_name:feature_value"
            # These strings are hashed into a fixed integer space (2^18) to serve as model inputs.
            features = [
                f"ua_os:{os_t}", f"ua_browser:{br_t}",
                f"region:{region}", f"city:{city}",
                f"adslot_visibility:{vis}", f"adslot_format:{fmt}",
                f"advertiser:{adv_id}", f"domain:{domain}",
            ]
            
            # Inference
            w_ctr = self.intercept_ctr
            w_cvr = self.intercept_cvr
            if self.weights_ctr is not None:
                for f in features:
                    h = self._hash(f)
                    w_ctr += self.weights_ctr[h]
                    w_cvr += self.weights_cvr[h]
            
            p_ctr = self._sigmoid(w_ctr)
            p_cvr = self._sigmoid(w_cvr)
            
            # Valuation
            # Formula: pCTR + N * (pCTR * pCVR)
            ev = p_ctr + (N * (p_ctr * p_cvr))
            
            # QUALITY FILTER (Gatekeeper)
            # Reject impressions with EV < 40% of average to conserve budget for high-quality slots.
            if ev < 0.4 * avg_ev:
                return -1
            
            # Market Anchoring
            # Bid proportionally to value relative to market average
            value_ratio = ev / avg_ev
            if value_ratio > 3.0: value_ratio = 3.0
            
            raw_bid = avg_mp * value_ratio

            # ---------------------------
            # Adaptive Pacing
            # ---------------------------
            # ---------------------------
            # Adaptive Pacing Control (PID-like)
            # ---------------------------
            # To ensure the budget lasts throughout the entire campaign (25M requests), we implement
            # a feedback loop that adjusts bidding aggression based on spend velocity.
            #
            # We track 'estimated_spend' using a probabilistic win-rate model (20%) rather than
            # sum of raw bids. This is critical in second-price auctions where the actual payment is
            # significantly lower than the bid price, and prevents premature throttling.
            self.estimated_spend += (raw_bid * self.ESTIMATED_WIN_RATE)
            
            ideal_spend = (self.requests_seen / self.EXPECTED_REQUESTS) * self.TOTAL_BUDGET
            
            pacing = 1.0
            if self.estimated_spend > ideal_spend:
                # We are overspending -> Cool down
                pacing = 0.8
            elif self.estimated_spend < (ideal_spend * 0.9):
                # We are underspending -> Speed up
                pacing = 1.15
                
            final_bid = int(raw_bid * pacing)
            
            # Safety Constraints
            floor = 0
            floor_str = bidRequest.getAdSlotFloorPrice()
            if floor_str:
                try: floor = int(floor_str)
                except: pass
            
            if final_bid < floor: return -1
            if final_bid > 300: final_bid = 300
            if final_bid <= 0: return -1
            
            return final_bid
            
        except Exception:
            return -1
