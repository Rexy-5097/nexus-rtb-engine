import threading
import time
import random
import logging
from src.distributed.budget_coordinator import BudgetCoordinator
from src.bidding.pacing import PacingController
# Mock engine for distributed test to avoid heavy ML loading
# In real distributed test we'd spawn processes, but threads are fine for logic check.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DistributedSim")

class MockBidderInstance(threading.Thread):
    def __init__(self, instance_id: str, coordinator: BudgetCoordinator, request_rate: int):
        super().__init__()
        self.instance_id = instance_id
        self.coordinator = coordinator
        self.request_rate = request_rate
        self.local_budget = 0.0
        self.running = True
        self.spent = 0.0
        self.bids_placed = 0

    def run(self):
        """Simulate bidding loop."""
        while self.running and self.coordinator.get_global_state()["remaining"] > 0:
            # 1. Check Local Budget
            if self.local_budget < 100:
                # Request chunk
                grant = self.coordinator.request_budget(self.instance_id, 1000.0)
                if grant == 0 and self.coordinator.get_global_state()["remaining"] <= 0:
                    break
                self.local_budget += grant
                
            # 2. Simulate Bid
            # Constant burn
            bid_cost = random.randint(1, 10)
            if self.local_budget >= bid_cost:
                self.local_budget -= bid_cost
                self.spent += bid_cost
                self.bids_placed += 1
            
            # Simulate latency
            time.sleep(1.0 / self.request_rate)

def run_distributed_simulation(instances=5, total_budget=50000):
    coordinator = BudgetCoordinator(total_budget)
    bidders = []
    
    logger.info(f"Starting distributed simulation with {instances} instances and ${total_budget} budget.")
    
    start_time = time.time()
    
    for i in range(instances):
        # Vary simulation speed
        rate = random.randint(50, 150) 
        b = MockBidderInstance(f"worker_{i}", coordinator, rate)
        bidders.append(b)
        b.start()
        
    # Monitor
    while any(b.is_alive() for b in bidders):
        state = coordinator.get_global_state()
        logger.info(f"Global Remaining: {state['remaining']:.2f}")
        if state["remaining"] <= 0:
            logger.info("Budget exhausted! Stopping instances.")
            for b in bidders:
                b.running = False
            break
        time.sleep(0.5)
        
    for b in bidders:
        b.join()
        
    duration = time.time() - start_time
    total_spent = sum(b.spent for b in bidders)
    
    logger.info("="*30)
    logger.info("DISTRIBUTED RESULT")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Total Budget: {total_budget}")
    logger.info(f"Total Spent: {total_spent}")
    logger.info(f"Overshoot: {total_spent - total_budget}")
    logger.info("="*30)
    
    return {
        "instances": instances,
        "duration": duration,
        "total_spent": total_spent,
        "overshoot": total_spent - total_budget
    }

if __name__ == "__main__":
    run_distributed_simulation()
