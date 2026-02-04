import zlib
import math
import pickle
import os

# Import user-provided interfaces
try:
    from BidRequest import BidRequest
    from Bidder import Bidder
except ImportError:
    pass


class Bid(Bidder):
    def __init__(self):
        # -------------------------------
        # Configuration
        # -------------------------------
        self.FEATURE_BITS = 18
        self.HASH_SPACE = 2 ** self.FEATURE_BITS

        # Expected volume and budget (used for pacing)
        self.EXPECTED_REQUESTS = 25_000_000
        self.TOTAL_BUDGET = 25_000_000

        self.requests_seen = 0
        self.cumulative_bid_value = 0.0

        # Advertiser N values
        self.n_map = {
            "1458": 0,
            "3358": 2,
            "3386": 0,
            "3427": 0,
            "3476": 10,
        }

        # Default market anchors (fallbacks)
        self.stats = {
            "1458": {"avg_mp": 69.50, "avg_ev": 0.000801},
            "3358": {"avg_mp": 92.37, "avg_ev": 0.001174},
            "3386": {"avg_mp": 77.24, "avg_ev": 0.000723},
            "3427": {"avg_mp": 81.51, "avg_ev": 0.000748},
            "3476": {"avg_mp": 79.62, "avg_ev": 0.000674},
        }

        # Model parameters
        self.weights_ctr = None
        self.weights_cvr = None
        self.intercept_ctr = 0.0
        self.intercept_cvr = 0.0

        # -------------------------------
        # Load trained model
        # -------------------------------
        try:
            model_path = os.path.join(os.path.dirname(__file__), "model_weights.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    data = pickle.load(f)

                if "ctr" in data:
                    self.weights_ctr = data["ctr"]["coef"].reshape(-1)
                    self.intercept_ctr = float(data["ctr"]["intercept"][0])

                if "cvr" in data:
                    self.weights_cvr = data["cvr"]["coef"].reshape(-1)
                    self.intercept_cvr = float(data["cvr"]["intercept"][0])

                # Prefer stats from training if present
                if "stats" in data and data["stats"]:
                    self.stats = data["stats"]
        except Exception:
            # Fail-safe: proceed with defaults
            pass

    # -------------------------------
    # Utility functions
    # -------------------------------
    def _hash(self, s: str) -> int:
        if not s:
            return 0
        return (zlib.adler32(s.encode("utf-8")) & 0xffffffff) % self.HASH_SPACE

    def _sigmoid(self, x: float) -> float:
        if x < -15:
            return 0.0000003
        if x > 15:
            return 0.9999997
        return 1.0 / (1.0 + math.exp(-x))

    def _norm(self, v):
        if v is None:
            return "unknown"
        s = str(v)
        return "unknown" if s == "" or s.lower() == "nan" else s

    def _parse_ua(self, ua):
        if not ua or ua == "unknown":
            return "unknown", "unknown"
        ua = ua.lower()

        # OS
        if "windows" in ua:
            os_t = "windows"
        elif "mac" in ua:
            os_t = "mac"
        elif "ios" in ua:
            os_t = "ios"
        elif "android" in ua:
            os_t = "android"
        elif "linux" in ua:
            os_t = "linux"
        else:
            os_t = "other"

        # Browser
        if "edge" in ua:
            br_t = "edge"
        elif "chrome" in ua:
            br_t = "chrome"
        elif "firefox" in ua:
            br_t = "firefox"
        elif "safari" in ua:
            br_t = "safari"
        elif "msie" in ua or "trident" in ua:
            br_t = "ie"
        elif "opera" in ua:
            br_t = "opera"
        else:
            br_t = "other"

        return os_t, br_t

    # -------------------------------
    # Core bidding logic
    # -------------------------------
    def getBidPrice(self, bidRequest: BidRequest) -> int:
        try:
            self.requests_seen += 1

            adv_id = str(bidRequest.getAdvertiserId())
            N = self.n_map.get(adv_id, 0)

            stat = self.stats.get(adv_id, {"avg_mp": 70.0, "avg_ev": 0.001})
            avg_mp = float(stat["avg_mp"])
            avg_ev = float(stat["avg_ev"])
            if avg_ev <= 1e-9:
                avg_ev = 0.0001

            # ---------------------------
            # Feature extraction
            # ---------------------------
            ua = bidRequest.getUserAgent()
            region = self._norm(bidRequest.getRegion())
            city = self._norm(bidRequest.getCity())
            domain = self._norm(bidRequest.getDomain())
            vis = self._norm(bidRequest.getAdSlotVisibility())
            fmt = self._norm(bidRequest.getAdSlotFormat())

            os_t, br_t = self._parse_ua(ua)

            features = [
                f"ua_os:{os_t}",
                f"ua_browser:{br_t}",
                f"region:{region}",
                f"city:{city}",
                f"adslot_visibility:{vis}",
                f"adslot_format:{fmt}",
                f"advertiser:{adv_id}",
                f"domain:{domain}",
            ]

            # ---------------------------
            # Inference
            # ---------------------------
            w_ctr = self.intercept_ctr
            w_cvr = self.intercept_cvr

            if self.weights_ctr is not None:
                for f in features:
                    h = self._hash(f)
                    w_ctr += self.weights_ctr[h]
                    w_cvr += self.weights_cvr[h]

            p_ctr = self._sigmoid(w_ctr)
            p_cvr = self._sigmoid(w_cvr)

            # Expected Value
            ev = p_ctr + (N * (p_ctr * p_cvr))

            if ev < 0.25 * avg_ev:
                return -1

            # ---------------------------
            # Market-anchored bidding
            # ---------------------------
            value_ratio = ev / avg_ev
            if value_ratio > 3.0:
                value_ratio = 3.0

            raw_bid = avg_mp * value_ratio

            # ---------------------------
            # Adaptive pacing
            # ---------------------------
            self.cumulative_bid_value += avg_mp
            ideal_spend = (self.requests_seen / self.EXPECTED_REQUESTS) * self.TOTAL_BUDGET
            actual_spend = self.cumulative_bid_value

            pacing = 1.0
            if actual_spend > ideal_spend:
                pacing = 0.8
            elif actual_spend < ideal_spend * 0.9:
                pacing = 1.15

            final_bid = int(raw_bid * pacing)

            # ---------------------------
            # Safety constraints
            # ---------------------------
            floor = 0
            floor_str = bidRequest.getAdSlotFloorPrice()
            if floor_str:
                try:
                    floor = int(floor_str)
                except Exception:
                    pass

            if final_bid < floor:
                return -1
            if final_bid > 300:
                final_bid = 300
            if final_bid <= 0:
                return -1

            return final_bid

        except Exception:
            return -1
