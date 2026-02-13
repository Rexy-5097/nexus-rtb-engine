# Changelog

## v2.0.0 – Economic Hardening Release

### Critical Safety Upgrades

- **Budget Cap Enforcement**: Added atomic, thread-safe tracking of `total_budget`, `remaining_budget`, and `spent_budget`. Circuit breaker engages immediately upon exhaustion.
- **Fail-Closed Model Loading**: Engine now refuses to answer bids if model artifacts are missing or fail integrity checks (Fail-Safe).
- **Adaptive Shading**: Implemented dynamic bid shading based on rolling win-rate observation (Target: 40%).
- **ROI Safety Checks**: Added guardrail ensuring `ExpectedValue` justifies the `RawBid` relative to market price.
- **Concurrency Hardening**: `PacingController` now uses `threading.Lock` for all state mutations.

### Features

- **Engine**: Expected Value (EV) valuation logic: $Bid = \alpha \cdot (pCTR \cdot V_{click} + pCVR \cdot V_{conv})$.
- **Pacing**: PID Controller integration for velocity smoothing.
- **Security**: `.npz` model loading (NumPy) preferred over Pickle.

### Breaking Changes

- `process()` now strictly returns `0` (No Bid) on all errors or safety violations.
- `EngineConfig` structure updated with strict budget fields.
