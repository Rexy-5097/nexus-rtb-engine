# System Architecture: Nexus-RTB Engine

## Overview

The Nexus-RTB Engine is a high-performance, real-time bidding system designed to respond to bid requests within **5 milliseconds**. It uses a dual Logistic Regression model to predict Click-Through Rate (CTR) and Conversion Rate (CVR), combined with an adaptive PID controller for budget pacing.

## High-Level Design

```mermaid
graph TD
    A[Ad Exchange] -->|Bid Request (JSON)| B(FastAPI Server)
    B --> C{Bidding Engine}

    subgraph Core Logic [src/bidding]
    C -->|1. Validate| D[Validator]
    C -->|2. Extract Features| E[Feature Extractor]
    E -->|Hash| F[Hasher (Adler32/SHA256)]
    C -->|3. Inference| G[LR Model (CTR/CVR)]
    C -->|4. Valuation| H[EV Calculator]
    C -->|5. Pacing| I[PID Controller]
    end

    G -->|Load Weights| J[(model_weights.pkl)]
    C -->|Bid Response| A
```

## Module Responsibilities

### 1. `src/bidding` (Core Domain)

- **`engine.py`**: The central coordinator. Orchestrates the flow from request to response.
- **`features.py`**: Handles feature extraction and the "hashing trick". Converts raw strings (User-Agent, Domain, etc.) into a sparse feature vector ($2^{18}$ dimensions).
- **`model.py`**: Manages model lifecycle. Loads weights safely and provides fallbacks.
- **`pacing.py`**: Implements a PID controller to smooth budget spend over time, preventing early exhaustion.
- **`config.py`**: Centralized configuration for hyperparameters, thresholds, and constraints.

### 2. `src/utils` (Shared)

- **`hashing.py`**: Provides consistent hashing implementations (Adler32/SHA256).
- **`validation.py`**: Input sanitization and safety guards.

### 3. `src/training` (Pipeline)

- **`train.py`**: A streaming, out-of-core training pipeline suitable for large datasets.
  - **Event Indexing**: Uses SQLite to index clicks/conversions.
  - **Streaming**: Processes logs in chunks to keep memory usage constant.
  - **SGD**: Uses online learning (`partial_fit`) to train models incrementally.

## Key Design Decisions

### The Hashing Trick

We map high-cardinality categorical features (like Domain or User-Agent) to a fixed-size vector space ($2^{18}$) using a hash function.

- **Pros**: Constant memory usage, no need for vocabulary lookup dictionaries, handles unseen features gracefully.
- **Cons**: Hash collisions can introduce noise (mitigated by large hash space).

### Adaptive Pacing (PID)

To ensure the budget lasts the entire day, we don't just bid blindly.

- **Error Signal**: `(Target Spend - Actual Spend)`
- **Control Output**: `Pacing Factor` (adjusts bid price).
- This creates a feedback loop that speeds up spending if we are behind, and slows down if we are ahead.

### Fail-Safe Engineering

- **Model Loading**: If the artifacts are missing/corrupt, the system falls back to conservative default intercepts (~1.8% CTR).
- **Input Validation**: All strings are truncated to prevent memory DoS.
- **Latency**: The critical path avoids I/O and complex allocations.
