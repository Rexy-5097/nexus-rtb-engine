import time
import statistics
import random
import psutil
import os
from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest

def generate_random_request(i):
    return BidRequest(
        bidId=f"bench_{i}",
        timestamp=str(int(time.time())),
        visitorId=f"v_{random.randint(1, 100000)}",
        userAgent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        ipAddress=f"192.168.1.{random.randint(1, 255)}",
        region=str(random.randint(0, 30)),
        city=str(random.randint(0, 300)),
        adExchange="1",
        domain=f"site-{random.randint(1, 500)}.com",
        url=f"http://site-{random.randint(1, 500)}.com/page",
        anonymousURLID="",
        adSlotID=f"slot_{random.randint(1, 5)}",
        adSlotWidth="300",
        adSlotHeight="250",
        adSlotVisibility="1",
        adSlotFormat="1",
        adSlotFloorPrice="10",
        creativeID="c1",
        advertiserId="1458",
        userTags="t1"
    )

def benchmark(n=10000):
    print(f"Initializing engine...")
    # Ensure model path is correct relative to execution
    model_path = os.path.join("src", "model_weights.pkl")
    # For benchmark we might need to mock if file doesn't exist, but let's assume it does or use the fallback
    engine = BiddingEngine(model_path=model_path)
    
    print(f"Generating {n} requests...")
    requests = [generate_random_request(i) for i in range(n)]
    
    print("Warming up...")
    for _ in range(100):
        engine.process(requests[0])
        
    print("Running benchmark...")
    latencies = []
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    for req in requests:
        t0 = time.perf_counter_ns()
        engine.process(req)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1_000_000.0) # ms
        
    end_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(n * 0.95)]
    p99 = sorted(latencies)[int(n * 0.99)]
    
    print("\n" + "="*30)
    print(" BENCHMARK RESULTS")
    print("="*30)
    print(f"Requests processed: {n}")
    print(f"Average Latency:    {avg:.4f} ms")
    print(f"P50 Latency:        {p50:.4f} ms")
    print(f"P95 Latency:        {p95:.4f} ms")
    print(f"P99 Latency:        {p99:.4f} ms")
    print("-" * 30)
    print(f"Memory Usage:       {end_mem:.2f} MB")
    print(f"Memory Growth:      {end_mem - start_mem:.2f} MB")
    print("="*30)
    
    if avg > 5.0:
        print("❌ FAILED: Average latency > 5ms")
        exit(1)
    else:
        print("✅ PASSED: Latency is within SLA")

if __name__ == "__main__":
    benchmark(10000)
