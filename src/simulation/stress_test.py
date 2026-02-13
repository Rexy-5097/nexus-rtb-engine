import logging
import time
import random
from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest
from src.simulation.replay import AuctionSimulator, generate_mock_stream

logger = logging.getLogger("StressTest")
logging.basicConfig(level=logging.INFO)

class StressTester:
    """
    Simulates high-load and drift scenarios.
    """
    
    def __init__(self):
        self.engine = BiddingEngine(model_path="src/model_weights.pkl")
        self.simulator = AuctionSimulator(self.engine)

    def run_drift_scenario(self, n_impressions=50000):
        """
        Simulate a market where CTR/CVR shifts drastically (e.g., Click Farm).
        Goal: Verify Pacing doesn't panic or overspend.
        """
        logger.info(f"Starting Drift Scenario: {n_impressions} impressions...")
        
        # 1. Normal Phase
        logger.info("Phase 1: Normal Traffic")
        for req, mp, clk, conv in generate_mock_stream(int(n_impressions * 0.3)):
            self.simulator.run_event(req, mp, clk, conv)
            
        # 2. Drift Phase (CTR spikes 5x)
        logger.info("Phase 2: CTR Spike (5x)")
        for req, mp, clk, conv in generate_mock_stream(int(n_impressions * 0.3)):
            # Force high CTR
            if random.random() < 0.05: # 5% CTR
                clk = True
            self.simulator.run_event(req, mp, clk, conv)

        # 3. Market Price Phase (Price Doubles)
        logger.info("Phase 3: Market Shock (Price 2x)")
        for req, mp, clk, conv in generate_mock_stream(int(n_impressions * 0.4)):
            mp *= 2
            self.simulator.run_event(req, mp, clk, conv)

        report = self.simulator.report()
        logger.info("Stress Test Complete.")
        return report

if __name__ == "__main__":
    tester = StressTester()
    res = tester.run_drift_scenario(10000)
    print(res)
