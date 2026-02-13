# Monitoring Guide

The Nexus-RTB Engine comes with a full-stack observability suite using **Prometheus** and **Grafana**.

## Architecture

- **Nexus-RTB**: Exposes metrics at `/metrics`.
- **Prometheus**: Scrapes `/metrics` every 5 seconds.
- **Grafana**: Visualizes metrics via pre-configured dashboards.

## Quick Start

Start the entire stack:

```bash
docker-compose up -d
```

Access the dashboards:

- **Grafana**: [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **App Health**: [http://localhost:8000/health](http://localhost:8000/health)

## Key Metrics

| Metric Name           | Type      | Description                            |
| --------------------- | --------- | -------------------------------------- |
| `rtb_requests_total`  | Counter   | Total number of bid requests received. |
| `rtb_latency_seconds` | Histogram | End-to-end request processing time.    |
| `rtb_bid_price`       | Histogram | Distribution of bid prices offered.    |
| `rtb_estimated_spend` | Gauge     | Total estimated daily spend.           |
| `rtb_errors_total`    | Counter   | Count of exceptions or failures.       |

## Dashboard Panels

1. **Requests Per Second**: Real-time traffic volume.
2. **Latency**: Average, P95, and P99 latency.
3. **Budget Burn Rate**: Visualization of `estimated_spend` vs time.
