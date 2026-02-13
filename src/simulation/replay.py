import logging
import time
import random
from typing import List, Dict, Any
from dataclasses import asdict

from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest
# Assume data loader or mock generator is available or implemented here
# For replay, we need a stream of (BidRequest, PayingPrice, Click, Conversion)

logger = logging.getLogger(__name__)

class AuctionSimulator:
    """
    Replays historical or synthetic bid requests against the Bidding Engine.
    Simulates a Second-Price Auction environment.
    """
    
    def __init__(self, engine: BiddingEngine):
        self.engine = engine
        self.stats = {
            "requests": 0,
            "bids": 0,
            "wins": 0,
            "spend": 0.0,
            "clicks": 0,
            "conversions": 0,
            "score": 0.0  # Clicks + N * Conversions
        }
        self.history = [] # Optional: Store per-request logs? Might be too big.

    def run_event(self, request: BidRequest, market_price: float, is_click: bool, is_conv: bool, n_value: float = 0.0):
        """
        Process a single auction event.
        """
        self.stats["requests"] += 1
        
        # 1. Get Bid
        response = self.engine.process(request)
        my_bid = response.bidPrice
        
        # 2. Auction Logic (Second Price)
        # We win if our bid >= market_price (payingprice in logs usually represents the price paid by the winner, 
        # or the floor if no one else bid. In datasets like iPinYou, 'payingprice' is the winning price.)
        # If we bid higher than 'payingprice', we assume we would have won at 'payingprice'.
        # Note: This is an approximation. Real counter-factual analysis is harder.
        
        if my_bid >= market_price:
            self.stats["wins"] += 1
            self.stats["bids"] += 1
            cost = market_price # Second price assumption
            self.stats["spend"] += cost
            
            # 3. Attribution
            # If we win, do we get the click/conversion?
            # In historical replay, we only know if a click happened for the *original* winner.
            # Use "Attribution Probability" or simple "Match" if we assume we are the original winner?
            # Standard backtest assumption: If historical data had a click, and we win, we get the click.
            # (Valid only if we assume our ad is as good as the historical winner).
            if is_click:
                self.stats["clicks"] += 1
                self.stats["score"] += 1
            if is_conv:
                self.stats["conversions"] += 1
                self.stats["score"] += n_value
        elif my_bid > 0:
            self.stats["bids"] += 1
            # Lost
            pass

    def report(self):
        """Return summary stats."""
        s = self.stats
        roi = s["score"] / s["spend"] if s["spend"] > 0 else 0
        win_rate = s["wins"] / s["requests"] if s["requests"] > 0 else 0
        ecpc = s["spend"] / s["clicks"] if s["clicks"] > 0 else 0
        ecpa = s["spend"] / s["conversions"] if s["conversions"] > 0 else 0
        
        return {
            "Total Requests": s["requests"],
            "Total Wins": s["wins"],
            "Win Rate": f"{win_rate:.2%}",
            "Total Spend": f"{s['spend']:.2f}",
            "Total Clicks": s["clicks"],
            "Total Conversions": s["conversions"],
            "eCPC": f"{ecpc:.2f}",
            "eCPA": f"{ecpa:.2f}",
            "ROI (Score/Spend)": f"{roi:.4f}",
            "Final Score": s["score"]
        }

# Mock Data Generator for testing without raw logs
def generate_mock_stream(n=1000):
    for i in range(n):
        req = BidRequest(
            bidId=f"sim_{i}", timestamp=str(int(time.time())), visitorId="v1", userAgent="Mozilla",
            ipAddress="1.1.1.1", region="1", city="1", adExchange="1", domain="test.com",
            url="http://test.com", anonymousURLID="", adSlotID="1", adSlotWidth="300",
            adSlotHeight="250", adSlotVisibility="1", adSlotFormat="1", adSlotFloorPrice="0",
            creativeID="c1", advertiserId="1458", userTags=""
        )
        # Synthetic market
        mp = random.randint(5, 250)
        click = random.random() < 0.005 # 0.5% CTR
        conv = click and (random.random() < 0.1) # 10% CVR given click
        yield req, mp, click, conv

if __name__ == "__main__":
    # Basic self-test
    engine = BiddingEngine(model_path="src/model_weights.pkl")
    sim = AuctionSimulator(engine)
    print("Running simulation on synthetic data...")
    for req, mp, clk, conv in generate_mock_stream(5000):
        sim.run_event(req, mp, clk, conv, n_value=10.0) # Assume N=10 for simplicity
    
    import json
    print(json.dumps(sim.report(), indent=2))
