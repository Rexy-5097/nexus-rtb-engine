# Nexus-RTB: High-Performance Real-Time Bidding Engine

[![CI Status](https://github.com/Rexy-5097/nexus-rtb-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Rexy-5097/nexus-rtb-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 1. Executive Summary

**Nexus-RTB** is a production-grade inference engine designed for programmatic advertising auctions. It processes bid requests in **<4ms** (P99), operates within a strict **512MB** memory footprint, and implements a mathematically rigorous bidding strategy anchored by **Logistic Regression** (CTR/CVR) and **PID-based Budget Pacing**.

Built with a "Clean Architecture" philosophy, it separates core domain logic from infrastructure concerns, enabling seamless deployment via **Docker** and **FastAPI**, with full-stack observability via **Prometheus** and **Grafana**.

---

## 2. System Architecture

The system follows a synchronous, blocking I/O model for the critical path to minimize context-switching overhead, delegating heavy lifting (feature hashing) to optimized C-extensions.

```mermaid
graph TD
    A[Ad Exchange] -->|Bid Request (JSON)| B(FastAPI Gateway)
    B --> C{Bidding Engine}

    subgraph "Core Domain [src/bidding]"
    C -->|1. Feature Hashing using MurmurHash3| D[Feature Extractor]
    C -->|2. Dot Product Inference| E[LR Model]
    C -->|3. Economic Valuation| F[EV Calculator]
    C -->|4. Budget Control| G[PID Controller]
    end

    subgraph "Infrastructure"
    M[(Model Registry)] -->|Signed Weights| E
    P[Prometheus] -.->|Scrape| B
    end

    C -->|Bid Response| A
```

### Key Components

- **Feature Extractor**: Implements the **Hashing Trick** ($2^{18}$ buckets) to handle high-cardinality categorical features (User-Agent, Domain) without dictionary lookups.
- **Inference Engine**: Dual Logistic Regression models (pCTR, pCVR) utilizing optimized sparse vector dot products.
- **Pacing Controller**: A Proportional-Integral-Derivative (PID) controller that regulates bid prices to smooth budget consumption over 24 hours.

---

## 3. Mathematical Strategy

### Valuation (Expected Value)

We determine the true economic value of an impression using the combined probability of click and conversion:

$$ EV = p(Click|Impression) \times p(Conversion|Click) \times Value\_{Conv} $$

### Adaptive Pacing

To prevent premature budget exhaustion, we apply a shading factor $\lambda$ derived from the PID controller:

$$ Bid\_{Price} = EV \times \lambda(t) $$

Where $\lambda(t)$ is adjusted dynamically based on the error between _Target Spend_ and _Actual Spend_.

### Market Anchoring

Bids are strictly bounded by floor prices and maximum caps to ensure rigorous margin safety:

$$ Bid*{Final} = \max(Floor, \min(Bid*{Price}, Cap\_{Max})) $$

---

## 4. Performance & Validation

| Metric               | Result        | Constraint | Status  |
| -------------------- | ------------- | ---------- | ------- |
| **Avg Latency**      | **3.9 μs**    | < 5.0 ms   | ✅ Pass |
| **P99 Latency**      | **5.3 μs**    | < 10.0 ms  | ✅ Pass |
| **Memory Footprint** | **45 MB**     | < 512 MB   | ✅ Pass |
| **Throughput**       | **~250k RPS** | -          | -       |

**Validation Reports:**

- [**Simulation Report**](docs/reports/SIMULATION_REPORT.md): Validated **2.5x ROI** improvement over baseline.
- [**Calibration Analysis**](docs/reports/CALIBRATION_REPORT.md): Confirmed model confidence alignment (Brier Score: 0.003).
- [**Stress Test**](docs/reports/STRESS_TEST_REPORT.md): Verified stability under 5x traffic spikes.

---

## 5. Deployment Guide

### Prerequisites

- Docker Engine
- Python 3.9+

### Quick Start

```bash
# 1. Build Container
docker build -t nexus-rtb .

# 2. Run Service (Exposes Port 8000)
docker run -p 8000:8000 --env-file .env nexus-rtb

# 3. Test Endpoint
curl -X POST http://localhost:8000/bid -d @tests/sample_request.json
```

### Monitoring

Access the observability stack:

- **Grafana**: `http://localhost:3000` (Dashboard: RTB Metrics)
- **Prometheus**: `http://localhost:9090`

---

## 6. Engineering Tradeoffs

### Why Logistic Regression over DNNs?

**Latency & Cost**. While Deep Neural Networks (DNNs) may capture complex non-linearities, they require ~20-50ms for inference on CPU. Nexus-RTB prioritizes extreme low latency (<5ms) to maximize auction participation rate.

### Why the Hashing Trick?

**Memory Safety**. Dictionary-based encoding explodes in memory with high-cardinality features (e.g., Millions of URLs). Hashing guarantees a fixed memory footprint ($2^{18}$ floats), essential for sidecar deployments.

### Why PID Pacing?

**Smoothness**. Simple "Throttle/No-Throttle" logic causes oscillation (bang-bang control). PID provides smooth, continuous adjustment of bid prices, ensuring stable delivery curves.

---

## 7. Engineering Highlights

- **Deterministic Inference**: 100% reproducible decision paths for identical inputs.
- **Fail-Closed Security**: Model loading enforces SHA256 signature verification.
- **Operational Maturity**: Structured logging, Prometheus instrumentation, and Health checks.
- **Distributed Ready**: Designed for stateless scaling with atomic budget coordination support.

---

_For internal documentation on the Distributed Architecture, refer to [DISTRIBUTED_DESIGN.md](DISTRIBUTED_DESIGN.md)._
