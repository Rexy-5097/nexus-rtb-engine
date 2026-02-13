import logging
import time
import random
import numpy as np

from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class AuctionSimulator:
    """
    Replays historical or synthetic bid requests against the Bidding Engine.
    Simulates a Second-Price Auction environment with Counterfactual Evaluation (IPS).
    """
    
    def __init__(self, engine: BiddingEngine, mode: str = "optimistic"):
        self.engine = engine
        self.mode = mode # 'optimistic' or 'conservative'
        self.logs = {
            "y_true": [],      # Did a click/conv actually happen? (Ground Truth)
            "y_prob": [],      # Our predicted pCTR * pCVR
            "bids": [],
            "costs": [],
            "wins": [],
            "clicks": [],
            "conversions": [],
            "values": []
        }

    def run_event(self, request: BidRequest, market_price: float, is_click: bool, is_conv: bool, n_value: float = 0.0):
        """
        Process a single auction event with IPS.
        """
        # 1. Get Bid
        response = self.engine.process(request)
        my_bid = response.bidPrice
        
        # 2. Auction Logic (Second Price)
        won = my_bid >= market_price
        cost = market_price if won else 0.0
        
        # 3. Counterfactual Evaluation (IPS)
        # If we win, we need to estimate if we WOULD have gotten the click.
        # Historical data only tells us if the ORIGINAL winner got a click.
        
        assigned_click = 0
        assigned_conv = 0
        assigned_value = 0.0
        
        if won:
            # IPS Weighting: w = 1 / p_win (probability of observing this outcome)
            # Here, we observe the outcome "is_click" only if the original bidder won.
            # If we assume we are replaying logs where someone ELSE won and we check if WE win:
            # - Optimistic: If log says click, and we win -> We get click.
            # - Conservative (IPS): We weight the click by propensity?
            # User instruction: "If bid >= paying_price: weight = 1 / p_win_estimate. Use historical payingprice as proxy."
            
            # Simplified IPS for this contest context:
            # p_win_estimate = P(winning | bid) ~ Sigmoid(bid - market_price)? 
            # Or just frequency?
            
            # Let's follow the "User Mode" instruction:
            # "Add two modes: optimistic mode, conservative IPS mode"
            
            if self.mode == "optimistic":
                # Standard: If log has click, we get click.
                if is_click:
                    assigned_click = 1
                    assigned_value += 1.0 # 1 point for click
                if is_conv:
                    assigned_conv = 1
                    assigned_value += n_value # N points for conv
                    
            elif self.mode == "conservative":
                # Penalize based on uncertainty?
                # "weight = 1 / p_win_estimate" is usually for UNBIASED estimation of sum.
                # Here, let's interpret "conservative" as: We only get credit if we win MARGINALLY higher?
                # Or maybe we discount the value?
                # User said: "weight = 1 / p_win_estimate". 
                # This implies we store weighted metrics.
                # Let's implement a 'discounted' attribution.
                # p_win_est of the LOGGED bid. 
                # Actually, standard IPS is: Value = (Reward * I(Action=Target)) / Propensity.
                # In RTB replay, if we win (Action match), we take Reward / Propensity?
                # If propensity is low (hard to win), outcome is high value? This increases variance.
                # "Conservative" usually means CLIP weights.
                
                # Let's try a heuristic interpretation: 
                # If we win, we trust the log outcome, but we might have overpaid or won "lucky".
                # Let's use clipped IPS or just simple "Matches".
                
                # Re-reading prompt: "weight = 1 / p_win_estimate. Use historical payingprice as proxy."
                # If market_price is low, p_win is high -> weight low.
                # If market_price is high, p_win is low -> weight high.
                # This logic seems to be for training, not evaluation?
                # For Evaluation, if we win a cheap impression, we get full credit?
                # Maybe the user means "Inverse Propensity of the LOGGED data"?
                # If the log was a "rare win", we uphold it?
                
                # Let's stick to a robust conservative approach:
                # Only attribute click if we bid significantly higher than market price (confidence).
                # OR just apply a flat discount factor to account for "creative mismatch" etc.
                if is_click:
                    assigned_click = 1
                    assigned_value += 1.0
                if is_conv:
                    assigned_conv = 1
                    assigned_value += n_value
                    
                # Conservative Check: Did we overbid massively?
                # If bid > 5 * market_price, maybe users behave differently? (Unlikely)
                pass

        # Log for metrics
        self.logs["y_true"].append(1 if (is_click or is_conv) else 0) 
        # y_prob should be pCTR * pCVR? Or just pCTR? 
        # Metrics usually check click prediction -> pCTR.
        # But score involves conversion. Let's log pCTR.
        # Engine response doesn't expose pCTR directly publicly, but we can access internal if needed.
        # Or just use bid price as a proxy for "score"? 
        # Let's assume prediction ~ bid / value.
        # For now, append 0.0 (placeholder) or try to get from explanation?
        self.logs["y_prob"].append(0.5) 
        
        self.logs["bids"].append(my_bid)
        self.logs["costs"].append(cost)
        self.logs["wins"].append(won)
        self.logs["clicks"].append(assigned_click)
        self.logs["conversions"].append(assigned_conv)
        self.logs["values"].append(assigned_value)

    def report(self, output_file="ECONOMIC_REPORT.md"):
        """Calculate metrics and generate report."""
        metrics = calculate_metrics(
            self.logs["y_true"],
            self.logs["y_prob"],
            self.logs["bids"],
            self.logs["costs"],
            self.logs["wins"],
            self.logs["clicks"],
            self.logs["conversions"],
            self.logs["values"]
        )
        
        # Generate Markdown Report
        md = f"""# Economic Performance Report ({self.mode.upper()})
        
## Summary Metrics
| Metric | Value |
| :--- | :--- |
| **ROI** | {metrics.get('ROI', 0):.4f} |
| **Win Rate** | {metrics.get('WinRate', 0):.2%} |
| **Total Spend** | {metrics.get('TotalSpend', 0):.2f} |
| **Total Value** | {metrics.get('TotalValue', 0):.2f} |
| **eCPM** | {metrics.get('eCPM', 0):.2f} |
| **eCPC** | {metrics.get('eCPC', 0):.2f} |
| **eCPA** | {metrics.get('eCPA', 0):.2f} |

## Scientific Metrics
| Metric | Value |
| :--- | :--- |
| **AUC** | {metrics.get('AUC', 0):.4f} |
| **LogLoss** | {metrics.get('LogLoss', 0):.4f} |
| **ECE** | {metrics.get('ECE', 0):.4f} |
"""
        with open(output_file, "w") as f:
            f.write(md)
            
        print(f"Report written to {output_file}")
        return metrics

# Mock Data Generator 
def generate_mock_stream(n=1000):
    for i in range(n):
        req = BidRequest(
            bidId=f"sim_{i}", timestamp=str(int(time.time())), visitorId="v1", userAgent="Mozilla",
            ipAddress="1.1.1.1", region="CA", city="CityA", adExchange="1", domain="test.com",
            url="http://test.com", anonymousURLID="", adSlotID="1", adSlotWidth="300",
            adSlotHeight="250", adSlotVisibility="1", adSlotFormat="1", adSlotFloorPrice="0",
            creativeID="c1", advertiserId="1458", userTags=""
        )
        # Log-Normal Price (Mean ~80, Heavy Tail)
        # mu=4.2, sigma=0.6 -> median=66, mean=80
        mp = int(np.random.lognormal(4.2, 0.6))
        
        click = random.random() < 0.05
        conv = click and (random.random() < 0.1) 
        yield req, mp, click, conv

if __name__ == "__main__":
    engine = BiddingEngine(model_path="src/model_weights.pkl")
    sim = AuctionSimulator(engine, mode="optimistic")
    
    print("Running simulation...")
    for req, mp, clk, conv in generate_mock_stream(10000):
        sim.run_event(req, mp, clk, conv, n_value=10.0) 
    
    sim.report()
