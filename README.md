# Nexus-RTB Engine

## 🚀 Overview

**Nexus-RTB** is a production-grade Real-Time Bidding (RTB) engine capable of processing bid requests in under **5ms**. It features a dual-model (CTR + CVR) inference engine, adaptive budget pacing via PID control, and a scalable, containerized architecture.

![Architecture Diagram](https://mermaid.ink/img/pako:eNqNVMtu2zAQ_BWCy0GAnnooCjRpLtoCbdGiQA89FJaFWEukSpGqRDMG_vfSciwnTdoWvYhYzszO7MzutEapVIFSFY_eWftmPe9-sMYY4y201m8sK1l5Y_kba2_fvdvQx_eXN3R1cXVx_fEiQdJIKaUQnBfC85LzUkhWCsFZKYQQrOSCFYITngvBScFZIQSjeSlEwXnOScEZKwQneMkF5y3mR_A8Z0UpeMFZITjP-C-s4LxkfHqW8UIwXkpeCMZLyW_g2f3d3dPNTV8cZ0_w_eGz-q_2iG-e4cf3t33xCN8__6r-qp_g-_fv-uIJfvzyqf6rZ_j-40dfnMA_X_5U_7UEfx4_9sUj_PDxs_qvHuD779_7og21t6wxtrHW8h-Wv7P23trHvtjD3-o_7W1fPMKPXz7Wf_UM33__1heP8P3nT_VfPcEPXz_3xSP8-PVT_VcKf6v_7IsT-PHrp_qv9vD91299cYK_1X_1xQn8dPuk_quHvtjDX-o_--IEfvr4Wf1XD_3x6wS__v4n_Pz1W_ihb7bgD18_9cX9E3z_9bv6jx7g-_fvffEIf3__rP6rZ_jxx0990R-_y3-v_6N-gh-_fO-Lf8OzvX16uq2z9vG2tvaxztr72tvHOmtva28f66y9rb19rLP2tvb2sc7a29rbxzprb2tvH-usva29fayz9rb29rHO2tva28c6a29rbx_rrL2tvX2ss_a29vaxztrb2tvHOmtva28f66y9rb19rLP2tvb2sc7a29rbxzprb2tvH-usvau9_TckJ87z)

## 🏗 Architecture

The system is built on a modular "Clean Architecture" pattern:

- **`src.bidding`**: Core domain logic (Inference, Valuation, Pacing, Config).
- **`src.utils`**: Shared utilities (Hashing, Validation).
- **`src.training`**: Streaming training pipeline.

## ✨ Key Features

- **Performance**: < 5ms P99 Latency.
- **Intelligence**: Combined CTR + CVR prediction using LR with the Hashing Trick ($2^{18}$ features).
- **Control**: PID-based pacing controller for smooth budget delivery.
- **Safety**: Fail-safe defaults, input sanitization, and strict timeouts.
- **Deployable**: Dockerized with FastAPI and Prometheus-ready hooks.

## 🛠 Project Structure

```bash
nexus-rtb-engine/
├── src/
│   ├── bidding/          # Core Engine Logic
│   │   ├── engine.py     # Orchestrator
│   │   ├── features.py   # Feature Extraction
│   │   ├── pacing.py     # PID Controller
│   │   └── config.py     # Configuration
│   ├── training/         # ML Pipeline
│   └── utils/            # Utilities
├── tests/                # Pytest Suite
├── benchmarks/           # Performance Scripts
├── deploy/               # Deployment Configs
├── docs/                 # Detailed Documentation
├── ARCHITECTURE.md       # System Design
├── MODEL_CARD.md         # Model Details
└── Dockerfile            # Container Definition
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- Docker

### 2. Run Tests

Validate the system logic:

```bash
pip install -r requirements.txt
pytest tests/
```

### 3. Run Benchmark

Verify performance SLA (<5ms):

```bash
python benchmarks/latency_benchmark.py
```

### 4. Build & Run

Deploy the container locally:

```bash
docker build -t nexus-rtb .
docker run -p 8000:8000 nexus-rtb
```

## 📚 Documentation

- [**System Architecture**](ARCHITECTURE.md)
- [**Deployment Guide**](DEPLOYMENT.md)
- [**Model Card**](MODEL_CARD.md)
- [**Security Audit**](docs/security_audit.md)
- [**Technical Report**](docs/project_report.md)

## 🤝 Contributing

This project is structured for high maintainability.

1. Fork the repo.
2. Create a feature branch.
3. Add tests for new logic.
4. Submit a Pull Request.

## 📄 License

MIT
