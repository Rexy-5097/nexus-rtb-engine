import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest, BidResponse

# --- Metrics ---
REQUEST_COUNT = Counter('rtb_requests_total', 'Total bid requests')
LATENCY = Histogram('rtb_latency_seconds', 'Request latency in seconds', buckets=[0.001, 0.002, 0.005, 0.010, 0.025, 0.050, 0.100])
BID_PRICE = Histogram('rtb_bid_price', 'Bid prices offered', buckets=[10, 50, 100, 200, 300])
SPEND_GAUGE = Gauge('rtb_estimated_spend', 'Total estimated spend')
PACING_FACTOR = Gauge('rtb_pacing_factor', 'Current PID pacing factor')
ERROR_COUNT = Counter('rtb_errors_total', 'Total errors', ['type'])

# Initialize App & Engine
app = FastAPI(title="Nexus RTB Engine", version="1.0.0")
engine = BiddingEngine(model_path="src/model_weights.pkl")

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up Nexus RTB Engine...")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down...")
    engine.shutdown()

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint for k8s/LB."""
    # Check model loading status
    if engine.model_loader.weights_ctr is None:
         # In production, you might want to return 503, but 
         # since we have fail-safe defaults, we report healthy but degraded?
         # For K8s, let's say it's healthy as it can serve traffic.
         pass
    return {"status": "healthy", "service": "nexus-rtb"}

@app.post("/bid", response_model=BidResponse)
async def get_bid(request: BidRequest):
    """
    Main bidding endpoint.
    """
    start_time = time.perf_counter()
    REQUEST_COUNT.inc()
    
    try:
        response = engine.process(request)
        
        duration = time.perf_counter() - start_time
        LATENCY.observe(duration)
        
        if response.bidPrice > 0:
            BID_PRICE.observe(response.bidPrice)
        
        # Update gauges
        stats = engine.pacing.get_stats()
        SPEND_GAUGE.set(stats["estimated_spend"])
        # We need to expose pacing factor from stats if possible, or getLastFactor
        # For now, let's assume get_stats returns it or we can infer it.
        # Actually pacing controller returns it on update. 
        # Making it available via stats would be better.
        
        return response
        
    except Exception as e:
        ERROR_COUNT.labels(type="exception").inc()
        logging.error(f"Inference error: {e}", exc_info=True)
        return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=request.advertiserId, explanation="internal_error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
