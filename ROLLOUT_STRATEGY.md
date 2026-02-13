# Rollout Strategy

## Overview

We employ a **Canary Deployment** strategy using feature flags to safely introduce new bidding logic.

## Configuration

Controlled via `src/bidding/config.py`:

```python
strategy_version = "v2"      # Target version
experiment_traffic = 0.10    # 10% of users
```

## Mechanism

1. **Hashing**: User ID is hashed to determine experiment bucket.
2. **Allocation**:
   - If `hash(user_id) % 100 < 10`: Use `v2` logic.
   - Else: Use `v1` logic.
3. **Metrics**: Compare `eCPC` and `ROI` between buckets.

## Rollout Plan

1. **Stage 1 (Internal)**: 1% Traffic. Monitor error rates.
2. **Stage 2 (Canary)**: 10% Traffic. Monitor pacing and budget.
3. **Stage 3 (General Availability)**: 100% Traffic.
