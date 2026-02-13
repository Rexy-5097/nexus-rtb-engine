import sys
import os
import time
import random
import numpy as np

# Add the submission_build folder to the path
sys.path.append(os.path.join(os.getcwd(), 'submission_build'))

try:
    from Bid import Bid
    from BidRequest import BidRequest
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def generate_mock_request(bid_id):
    req = BidRequest()
    req.setBidId(str(bid_id))
    req.setTimestamp(str(int(time.time())))
    req.setVisitorId(f"visitor_{random.randint(1000, 9999)}")
    req.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    req.setIpAddress("192.168.1.1")
    req.setRegion(random.choice(["1", "2", "3", "unknown"]))
    req.setCity(random.choice(["10", "20", "30", "unknown"]))
    req.setAdExchange("1")
    req.setDomain("example.com")
    req.setUrl("https://example.com/page")
    req.setAdSlotVisibility(random.choice(["1", "2", "unknown"]))
    req.setAdSlotFormat(random.choice(["1", "2", "unknown"]))
    req.setAdSlotFloorPrice(str(random.randint(10, 100)))
    req.setAdvertiserId(random.choice(["1458", "3358", "3386", "3427", "3476"]))
    return req

def run_benchmark(n_requests=10000):
    bidder = Bid()
    
    # Warm up
    for i in range(10):
        req = generate_mock_request(i)
        bidder.getBidPrice(req)
        
    latencies = []
    
    start_bench = time.perf_counter()
    for i in range(n_requests):
        req = generate_mock_request(i)
        
        start = time.perf_counter()
        bidder.getBidPrice(req)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000) # Convert to ms
        
    total_time = time.perf_counter() - start_bench
    
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"Benchmark Results (N={n_requests}):")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Latency: {avg_latency:.4f}ms")
    print(f"  P99 Latency: {p99_latency:.4f}ms")
    print(f"  Requests/Sec: {n_requests/total_time:.2f}")

    if avg_latency < 5.0:
        print("Latency Requirement (Avg < 5ms): PASSED")
    else:
         print("Latency Requirement (Avg < 5ms): FAILED")

if __name__ == "__main__":
    run_benchmark()
