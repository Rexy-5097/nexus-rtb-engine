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
            1. Pacing Circuit Breaker (Budget Check)
            2. Feature Extraction
            3. Inference (CTR/CVR)
            4. Valuation (EV = pCTR * Vc + pCVR * Vconv)
            5. ROI Guard (CPA check)
            6. Pacing Alpha application
            7. Final Constraints (Floor, Max, Min)
        """
        start_time = time.perf_counter()
        adv_id = str(request.advertiserId)
        
        try:
            # 1. Circuit Breaker (Hard Budget Cap & Surge)
            if not self.pacing.can_bid():
                return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="pacing_limited")

            # 2. Feature Extraction
            hashed_features = self.feature_extractor.extract(request)
            
            # 3. Inference
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

            # 4. Valuation (Expected Value)
            # EV = p(Click) * Value_Click + p(Conversion) * Value_Conversion
            # Note: p(Conversion) usually means p(Conversion|Impression) = pCTR * pCVR
            p_conv_imp = p_ctr * p_cvr
            
            ev = (p_ctr * config.value_click) + (p_conv_imp * config.value_conversion)
            
            # 5. ROI Guard (CPA Check)
            # Predicted CPA = Bid / p(Conversion). 
            # If (Cost / Conversion) > Max_CPA, don't bid.
            # We estimate Cost ~ Bid Price (First Price) or Market Price (Second Price).
            # Let's be conservative and use EV as proxy for willingness to pay.
            if p_conv_imp > 0:
                predicted_cpa = ev / p_conv_imp
                if predicted_cpa > config.max_cpa:
                     return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="roi_guard_cpa")
            
            # 6. Pacing & Shadow Bidding
            # We update pacing with the *intended* spend (EV) to regulate velocity
            alpha = self.pacing.update(ev)
            
            # Bid Shading / Alpha
            final_bid_float = ev * alpha
            
            # 7. Safety Checks & Constraints
            floor = 0.0
            try:
                floor = float(request.adSlotFloorPrice)
            except (ValueError, TypeError):
                pass
                
            final_bid = int(final_bid_float)
            
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
            # Fail Closed
            return BidResponse(bidId=request.bidId, bidPrice=0, advertiserId=adv_id, explanation="internal_error")

    def _sigmoid(self, x: float) -> float:
        """Stable sigmoid function."""
        if x < -15: return 0.0000003
        if x > 15: return 0.9999997
        return 1.0 / (1.0 + math.exp(-x))

    def shutdown(self):
        """Cleanup resources."""
        pass
