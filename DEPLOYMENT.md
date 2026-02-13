# Deployment Guide: Nexus-RTB Engine

This guide details how to build, deploy, and scale the Nexus RTB Engine in a production environment.

## Prerequisites

- Docker (20.10+)
- Python 3.9+ (for local testing)
- AWS/GCP account (for cloud deployment)

## Building the Container

The engine is packaged as a Docker container, optimized for minimal size and fast startup.

```bash
docker build -t nexus-rtb:latest .
```

## Running Locally

To run the engine locally on port 8000:

```bash
docker run -p 8000:8000 nexus-rtb:latest
```

### Health Check

Verify the service is running:

```bash
curl http://localhost:8000/health
# Output: {"status": "healthy", "service": "nexus-rtb"}
```

### Test Bidding

Send a sample bid request:

```bash
curl -X POST "http://localhost:8000/bid" \
     -H "Content-Type: application/json" \
     -d '{
           "bidId": "test_1",
           "timestamp": "1234567890",
           "visitorId": "visitor_1",
           "userAgent": "Mozilla/5.0...",
           "ipAddress": "192.168.1.1",
           "region": "15",
           "city": "8",
           "adExchange": "1",
           "domain": "example.com",
           "url": "http://example.com/page",
           "anonymousURLID": "",
           "adSlotID": "slot_1",
           "adSlotWidth": "300",
           "adSlotHeight": "250",
           "adSlotVisibility": "1",
           "adSlotFormat": "1",
           "adSlotFloorPrice": "50",
           "creativeID": "creative_1",
           "advertiserId": "1458",
           "userTags": "tag_1"
         }'
```

## Scalability & Performance

### Vertical Scaling (Resources)

- **CPU**: The inference path is CPU-bound (hashing + dot product). Recommended: 1 vCPU per instance.
- **Memory**: The model is lightweight (<50MB). However, Python overhead requires ~150-250MB per worker.
  - **Minimum**: 256MB RAM
  - **Recommended**: 512MB RAM

### Horizontal Scaling (K8s / ECS)

The engine is **stateless**. The pacing controller uses a local PID loop, which is an approximation in distributed settings.

- **Load Balancing**: Use a standard Layer 7 Load Balancer (ALB / Ingress).
- **Replica Count**: Scale based on CPU utilization (target 60%).
- **Pacing Note**: In a distributed setup (`N` replicas), the `total_budget` in `config.py` effectively becomes `N * budget` if not synchronized.
  - **Production Mitigation**: Divide the daily budget by `N` replicas, or implement a centralized budget coordinator (e.g., Redis).

## Production Tuning

### Uvicorn Workers

For maximum concurrency on a multi-core instance, run Uvicorn with multiple workers:

```bash
# In Dockerfile or command override
uvicorn deploy.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Monitoring

Integrate with Prometheus/Grafana. The `deploy/app.py` wrapper contains hooks for:

- Request Count (`rtb_requests_total`)
- Error Rate (`rtb_errors_total`)
- Latency Histogram (`rtb_latency_seconds`)
- Bid Price Distribution (`rtb_bid_price`)
- Estimated Spend (`rtb_estimated_spend`)

See [MONITORING.md](MONITORING.md) for full setup instructions using Docker Compose.

### Model Integrity

Ensure `model_weights.pkl.sig` is deployed alongside `model_weights.pkl`. The engine will **fail to load** the model if the signature is missing or invalid.
