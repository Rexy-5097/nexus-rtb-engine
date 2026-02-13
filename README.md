# Nexus-RTB Engine

![CI Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Coverage](https://img.shields.io/badge/coverage-94%25-green)
![License](https://img.shields.io/badge/license-MIT-grey)

**Nexus-RTB** is a high-performance Real-Time Bidding (RTB) engine designed for microsecond-latency ad auctions. It implements advanced economic safety mechanisms, including Expected Value (EV) bidding, PID-based pacing control, and hard budget enforcement, making it suitable for distributed production environments.

> **Production Note**: This repository represents a reference implementation of a sidecar bidding agent compatible with OpenRTB 2.5 standards.

---

## 🏗 System Architecture

The engine follows a **Clean Architecture** pattern, strictly separating I/O (FastAPI) from core bidding logic. It is designed to run as a stateless container (Kubernetes Pod) with a sidecar architecture.

```mermaid
graph TD
    User([ SSP / Ad Exchange ]) -->|OpenRTB Request| API[FastAPI Gateway]

    subgraph "Core Engine (Stateless)"
        API -->|Validate| Schema[Pydantic Schema]
        Schema -->|Extract| Feat[Feature Hashing (Murmur3/Adler32)]
        Feat -->|Infer| Model[Logistic Regression (Sparse)]

        Model -->|pCTR / pCVR| Valuation{Valuation Logic}
        Valuation -->|EV Calculation| Pacer[PID Pacing Controller]

        Pacer -.->|Feedback Loop| Budget[Budget State (Thread-Safe)]
    end

    Pacer -->|Bid Price| API
    API -->|Bid Response| User
```

---

## 💰 Economic Safety & Bidding Logic

Nexus-RTB prioritizes **financial safety** and **economic rationality** over raw volume.

### 1. expected Value (EV) Bidding

We use a dual-prediction model to estimate the unified value of an impression:

$$
Bid = \alpha \cdot (pCTR \cdot V_{click} + pCVR \cdot V_{conv})
$$

Where:

- $pCTR$: Probability of Click (Logistic Regression)
- $pCVR$: Probability of Conversion conditional on Click
- $\alpha$: Dynamic pacing factor (Bid Shading) controlled by the PID loop.
- $V_{click} / V_{conv}$: Configured base values.

### 2. PID Pacing Controller

To prevent budget exhaustion and dampen market shocks, we use a closed-loop **PID Controller**:

- **Proportional (P)**: Reacts to immediate spend velocity divergence.
- **Integral (I)**: Corrects long-term under/over-delivery.
- **Derivative (D)**: Dampens sudden spikes in market price (e.g., 2x shock).

### 3. Hard Budget Enforcement

The engine implements an atomic "Circuit Breaker" to guarantee budget compliance:

- **Daily Hard Cap**: \$25,000,000 (Global limit)
- **Hourly Soft Cap**: \$2,000,000 (Prevents early exhaustion)
- **Surge Protection**: \$50,000/minute (Dampens DDOS/Bot traffic)

---

## 🛡 Security & Reliability

| Feature | Implementation | Benefit |
| men | men | men |
| **Fail-Closed** | `try-except` blocks return `bidPrice=0` | Prevents "Zombie Bidding" on internal error. |
| **Model Integrity** | `sha256` signature verification | Prevents loading tampered/malicious model artifacts. |
| **Safe Loading** | `numpy.load` (allow_pickle=False) | Mitigates RCE risks associated with Python `pickle`. |
| **ROI Guard** | `if predicted_CPA > max_cpa: bid=0` | Prevents bidding on low-quality/high-cost inventory. |

---

## 🚀 Performance Benchmarks

Benchmarks run on `c5.2xlarge` (8 vCPU, 16GB RAM):

| Metric | Result | Target |
| men | men | men |
| **Avg Latency** | 1.2ms | < 5ms |
| **P99 Latency** | 3.8ms | < 10ms |
| **Throughput** | 12k QPS | > 10k QPS |
| **Memory** | 140MB | < 512MB |

---

## 🛠 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start Service
uvicorn deploy.app:app --host 0.0.0.0 --port 8000
```

### Docker Production

```bash
docker build -t nexus-rtb:v2.0.0 .
docker run -p 8000:8000 --env-file .env.prod nexus-rtb:v2.0.0
```

---

## 📦 Versioning Strategy

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **Major (v2.0.0)**: New Bidding Logic (EV), Safety Overhaul, Breaking Config Changes.
- **Minor (v1.1.0)**: New Features (e.g., new model features), Backward Compatible.
- **Patch (v1.0.1)**: Bug Fixes, Doc Updates.

---

## ⚠️ Risk Mitigation

- **Cold Start**: The PID controller starts with a conservative $\alpha=0.1$ and slowly ramps up.
- **Market Shocks**: If market prices double (2x shock), the **Derivative** term in the PID controller will aggressively reduce bids to prevent overpayment.
- **Model Drift**: Offline calibration monitoring is required (see `CALIBRATION_REPORT.md` for details).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
