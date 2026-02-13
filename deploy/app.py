import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest, BidResponse

# Convert internal schema to Pydantic if needed, or use dataclasses directly
# FastAPI supports dataclasses.

# Initialize App & Engine
app = FastAPI(title="Nexus RTB Engine", version="1.0.0")
engine = BiddingEngine(model_path="src/model_weights.pkl")

# Metrics
request_count = 0
error_count = 0

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up Nexus RTB Engine...")
    # Pre-warm functionality if needed

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down...")
    engine.shutdown()

@app.get("/health")
async def health_check():
    """Health check endpoint for k8s/LB."""
    # Add logic to check model loaded status
    return {"status": "healthy", "service": "nexus-rtb"}

@app.post("/bid", response_model=BidResponse)
async def get_bid(request: BidRequest):
    """
    Main bidding endpoint.
    Expects a JSON payload matching BidRequest schema.
    """
    start_time = time.perf_counter()
    global request_count, error_count
    request_count += 1
    
    try:
        # Convert Pydantic/Dict to dataclass if simple injection doesn't work automatically 
        # But FastAPI with dataclasses works fine usually.
        # Actually BidRequest is a dataclass.
        
        response = engine.process(request)
        
        latency = (time.perf_counter() - start_time) * 1000
        # Log latency (use structured logging in prod)
        if latency > 5:
            logging.warning(f"High latency: {latency:.2f}ms")
            
        return response
        
    except Exception as e:
        error_count += 1
        logging.error(f"Inference error: {e}", exc_info=True)
        # Fail-safe empty bid
        return BidResponse(bidId=request.bidId, bidPrice=-1, advertiserId=request.advertiserId, explanation="internal_error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
