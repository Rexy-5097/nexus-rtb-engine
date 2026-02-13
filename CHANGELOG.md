# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-13

### Added

- **CI/CD**: GitHub Actions workflow for automated testing and latency benchmarking.
- **Monitoring**: Full Prometheus + Grafana stack (`docker-compose.yml`) with pre-provisioned dashboards.
- **Model Integrity**: SHA256 signature verification for model weights (`src/utils/crypto.py`).
- **Telemetry**: Instrumented `deploy/app.py` with `prometheus_client`.
- **Documentation**: New `MONITORING.md`, `MODEL_INTEGRITY.md`, updated `ARCHITECTURE.md`.

### Changed

- **Architecture**: Refactored monolithic code into modular `src/bidding`, `src/utils`.
- **Safety**: Replaced raw pickle loading with signature-verified loading.
- **Performance**: Optimized feature extraction; verified <4µs latency.

### Security

- Added `scripts/sign_model.py` for generating model signatures.
- Enforced strict model loading policy (Fail-Closed).
