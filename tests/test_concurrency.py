import threading
import pytest
import time
from src.bidding.pacing import PacingController
from src.bidding.config import config

def test_atomic_reservation_limit():
    """
    Stress test for TOCTOU race condition.
    32 threads attempting to reserve 100 budget each.
    Total budget = 1000.
    Should allow exactly 10 reservations, not more.
    """
    pacer = PacingController()
    # Mock budget
    pacer._total_budget = 1000.0
    pacer._spent_budget = 0.0
    
    # Each thread tries to reserve 100
    threads = []
    results = []
    
    def worker():
        success = pacer.reserve_budget(100.0)
        results.append(success)
        
    for _ in range(32):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    success_count = sum(1 for r in results if r)
    
    # With 1000 budget and 100 per reservation, max 10 successes.
    # If race condition exists, might be > 10.
    assert success_count == 10
    assert pacer._spent_budget == 1000.0

def test_refund_logic():
    pacer = PacingController()
    pacer._total_budget = 1000.0
    
    assert pacer.reserve_budget(100.0)
    assert pacer._spent_budget == 100.0
    
    pacer.refund_budget(100.0)
    assert pacer._spent_budget == 0.0
    # Should be able to reserve again
    assert pacer.reserve_budget(100.0)

def test_hourly_cap_concurrency():
    pacer = PacingController()
    pacer._total_budget = 10000.0
    
    from dataclasses import replace
    # Replace the configuration on the pacer instance
    pacer.conf = replace(pacer.conf, max_hourly_spend=500.0)
    
    threads = []
    results = []
    
    # 20 threads * 50 = 1000 > 500 cap
    def worker():
        success = pacer.reserve_budget(50.0)
        results.append(success)
        
    for _ in range(20):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    success_count = sum(1 for r in results if r)
    assert success_count == 10 # 10 * 50 = 500
    assert pacer._hourly_spend == 500.0
