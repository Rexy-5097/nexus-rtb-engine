import math
import logging
from typing import Optional

from src.bidding.schema import BidRequest, BidResponse
from src.bidding.config import config
from src.bidding.features import FeatureExtractor
from src.bidding.pacing import PacingController
from src.bidding.model import ModelLoader
from src.utils.validation import Validator

logger = logging.getLogger(__name__)

class BiddingEngine:
    """
    Core RTB Engine.
    Orchestrates feature extraction, inference, valuation, quality gating, and pacing.
    """
    
    def __init__(self, model_path: str = "src/model_weights.pkl"):
        self.model_loader = ModelLoader(model_path)
        self.pacing = PacingController()
        
    def _sigmoid(self, x: float) -> float:
        """Stable sigmoid function."""
        if x < -15: return 0.0000003
        if x > 15: return 0.9999997
        return 1.0 / (1.0 + math.exp(-x))

    def process(self, request: BidRequest) -> BidResponse:
        """
        Process a bid request and decide whether to bid and at what price.
        """
        try:
            adv_id = Validator.sanitize_string(request.advertiserId)
            
            # --- 1. Feature Extraction ---
            # We must use FeatureExtractor but currently hashing logic is inside Bid.py
            # The refactor moves it to FeatureExtractor which uses Hasher.
            # However, FeatureExtractor.extract needs to return indices.
            # Wait, FeatureExtractor.extract creates hash indices.
            
            # Let's fix FeatureExtractor import usage.
            # We'll use the feature extraction logic directly here or via the helper.
            # But wait, FeatureExtractor.extract returns hashed_features list.
            
            # Let's verify FeatureExtractor is imported correctly in this mock code.
            # Yes.
            
            # Retrieve Model Weights
            # (Optimization: Local var access is faster)
            w_ctr_base = self.model_loader.intercept_ctr
            w_cvr_base = self.model_loader.intercept_cvr
            weights_ctr = self.model_loader.weights_ctr
            weights_cvr = self.model_loader.weights_cvr
            
            # --- 2. Inference ---
            w_ctr = w_ctr_base
            w_cvr = w_cvr_base
            
            # Helper to get features and apply weights
            # To avoid implementing FeatureExtractor logic twice, we use the class approach
            # But the hashing logic is state-less, so we can likely inline or call static.
            
            # Let's assume we implement the extraction logic here for speed or call the helper?
            # Calling helper method is cleaner.
            
            # We need to adapt FeatureExtractor to take the request object.
            # The request object needs to be mapped from dict if it comes from JSON, 
            # or passed as object if internal.
            # Here we assume 'request' is a BidRequest dataclass.
            
            # Extract features (hashing)
            # This is slightly inefficient if we create list then iterate.
            # But for readability and modularity -> src/bidding/features.py
            
            # Custom implementation for speed to match original logic exactly:
            # Original: _hash(f)
            # New: Hasher.adler32_hash(f)
            
            # Let's rewrite the feature extraction loop inside engine for performance
            # or rely on the FeatureExtractor class which does `return hashed_features`.
            
            # Feature Extraction
            # We need the exact features as `Bid.py`.
            # ua = request.userAgent ...
            
            # Let's stick to using the Features module feature extraction:
            # But... FeatureExtractor.extract takes `BidRequest` which is good.
            
            hashed_features = FeatureExtractor.extract(request)
            
            if weights_ctr is not None:
                for h in hashed_features:
                    w_ctr += weights_ctr[h]
                    w_cvr += weights_cvr[h]
            
            p_ctr = self._sigmoid(w_ctr)
            p_cvr = self._sigmoid(w_cvr)
            
            # --- 3. Valuation & Strategy ---
            N = config.model.n_map.get(adv_id, 0)
            ev = p_ctr + (N * p_ctr * p_cvr)
            
            stats = self.model_loader.get_stats(adv_id)
            avg_ev = stats["avg_ev"]
            avg_mp = stats["avg_mp"]
            
            # Quality Gate
            if ev < config.quality_threshold * avg_ev:
                return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=adv_id, explanation="quality_gate")
            
            # Market Anchoring
            value_ratio = ev / avg_ev
            if value_ratio > config.max_market_ratio:
                value_ratio = config.max_market_ratio
            
            raw_bid = avg_mp * value_ratio
            
            # --- 4. Pacing ---
            pacing_factor, _ = self.pacing.update(raw_bid)
            final_bid = int(raw_bid * pacing_factor)
            
            # --- 5. Safety Checks ---
            floor = Validator.parse_floor_price(request.adSlotFloorPrice)
            
            if final_bid < floor:
                return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=adv_id, explanation="below_floor")
            
            if final_bid > config.max_bid_price:
                final_bid = config.max_bid_price
                
            if final_bid <= config.min_bid_price:
                return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=adv_id, explanation="below_min_bid")

            return BidResponse(bidId=request.bidId, bidPrice=final_bid, advertiserId=adv_id)

        except Exception as e:
            logger.error(f"Error processing bid {request.bidId}: {e}", exc_info=True)
            return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=request.advertiserId, explanation="error")

    def shutdown(self):
        """Cleanup resources."""
        pass
