import math
import logging
import time
from typing import Optional

from src.bidding.features import FeatureExtractor
from src.bidding.model import ModelLoader
from src.bidding.pacing import PacingController
from src.bidding.schema import BidRequest, BidResponse
from src.bidding.config import config

# Configure structured logging
logger = logging.getLogger(__name__)

class BiddingEngine:
    """
    Core orchestration layer for Real-Time Bidding.
    
    Responsibilities:
        1. Feature Extraction (Hashing Trick)
        2. Inference (CTR/CVR Prediction)
        3. Valuation (Expected Value Calculation)
        4. Pacing (PID Budget Control)
        5. Response Generation
        
    Attributes:
        model_loader (ModelLoader): Manages model weights and safe loading.
        feature_extractor (FeatureExtractor): Converts raw requests to sparse vectors.
        pacing (PacingController): Manages budget spend velocity.
    """

    def __init__(self, model_path: str = "src/model_weights.pkl"):
        """
        Initialize the Bidding Engine.

        Args:
            model_path (str): Path to the pickled model artifacts.
        """
        logger.info(f"Initializing BiddingEngine with model at {model_path}")
        self.model_loader = ModelLoader(model_path)
        self.feature_extractor = FeatureExtractor()
        self.pacing = PacingController()

    def process(self, request: BidRequest) -> BidResponse:
        """
        Process a single BidRequest and return a BidResponse.

        Execution Flow:
            1. Validate Input
            2. Extract Features -> Sparse Vector
            3. Inference -> pCTR, pCVR
            4. Calculate Expected Value (EV)
            5. Apply Pacing Factor -> Bid Price
            6. Construct Response

        Args:
            request (BidRequest): The incoming bid request.

        Returns:
            BidResponse: The decision (bid price, advertiser ID, etc.).
        """
        start_time = time.perf_counter()
        
        try:
            # 1. Feature Extraction
            hashed_features = self.feature_extractor.extract(request)
            
            # 2. Inference (Logistic Regression)
            # Dot product: w . x + b
            # Optimization: We iterate only over non-zero features (Sparse)
            
            # CTR
            w_ctr = self.model_loader.intercept_ctr
            if self.model_loader.weights_ctr is not None:
                for h in hashed_features:
                    w_ctr += self.model_loader.weights_ctr[h]
            p_ctr = self._sigmoid(w_ctr)
            
            # CVR
            w_cvr = self.model_loader.intercept_cvr
            if self.model_loader.weights_cvr is not None:
                for h in hashed_features:
                    w_cvr += self.model_loader.weights_cvr[h]
            p_cvr = self._sigmoid(w_cvr)

            # 3. Valuation & Strategy
            adv_id = str(request.advertiserId)
            # Fetch advertiser-specific value multiplier (N-map)
            # Default to 0 if unknown
            N = config.model.n_map.get(adv_id, 0)
            
            # EV = p(Click) * p(Conv|Click) * Value + p(Click) * Value_Click?
            # Simplified EV model: pCTR * (1 + N * pCVR) ? 
            # Or standard: p(Conv) * Value_Conv. p(Conv) = pCTR * pCVR.
            # Let's assume the formula is: Score = pCTR + N * pCTR * pCVR
            # Or simplified for hackathon: EV = pCTR * pCVR * 1000 (CPM)
            
            # Reverting to original logic found in codebase:
            # ev = p_ctr + (N * p_ctr * p_cvr)
            ev = p_ctr + (N * p_ctr * p_cvr)
            
            # 4. Market Awareness
            stats = self.model_loader.get_stats(adv_id)
            avg_ev = stats.get("avg_ev", 0.001)
            avg_mp = stats.get("avg_mp", 50.0)
            
            # Quality Gate
            if ev < config.quality_threshold * avg_ev:
                return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="quality_gate")
            
            # Market Anchoring ranges
            value_ratio = ev / avg_ev if avg_ev > 0 else 1.0
            if value_ratio > config.max_market_ratio:
                value_ratio = config.max_market_ratio
            
            # Base Bid 
            raw_bid = avg_mp * value_ratio
            
            # 5. Pacing
            pacing_factor, _ = self.pacing.update(raw_bid)
            final_bid = int(raw_bid * pacing_factor)
            
            # 6. Safety Checks & Constraints
            floor = 0
            try:
                floor = float(request.adSlotFloorPrice)
            except (ValueError, TypeError):
                pass
            
            if final_bid < floor:
                return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="below_floor")
            
            if final_bid > config.max_bid_price:
                final_bid = config.max_bid_price
                
            if final_bid <= config.min_bid_price:
                 return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="below_min_bid")

            latency_ms = (time.perf_counter() - start_time) * 1000
            return BidResponse(bidId=request.bidId, bidPrice=final_bid, advertiserId=adv_id, explanation=f"ok_lat={latency_ms:.3f}ms")

        except Exception as e:
            logger.error(f"Error processing bid {request.bidId}: {e}", exc_info=True)
            return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=str(request.advertiserId), explanation="internal_error")

    def _sigmoid(self, x: float) -> float:
        """Stable sigmoid function."""
        if x < -15: return 0.0000003
        if x > 15: return 0.9999997
        return 1.0 / (1.0 + math.exp(-x))

    def shutdown(self):
        """Cleanup resources."""
        pass
