# Model Card: Nexus-RTB v1.0

## Model Details

- **Model Date**: February 2026
- **Model Type**: Dual Logistic Regression (Sparse)
  - `model_ctr`: Predicts Click-Through Probability ($p(Click|Imp)$)
  - `model_cvr`: Predicts Conversion Probability ($p(Conv|Click)$)
- **Framework**: Scikit-Learn (SGDClassifier with `log_loss`)
- **License**: MIT

## Intended Use

- **Primary Use Case**: Real-Time Bidding (RTB) valuation.
- **Constraints**:
  - Latency < 50$\mu$s per inference.
  - Memory < 100MB per model instance.
  - Features must be hashed (no dictionary lookups).

## Training Data

- **Dataset**: iPinYou (Sampled) / Synthetic Validation Set.
- **Preprocessing**:
  - **Hashing**: All categorical features mapped to $2^{18}$ buckets using `MurmurHash3` (or similar).
  - **Normalization**: Numerical features (if any) are min-max scaled.
- **Split**: 80% Train, 10% Validation, 10% Test (Time-based split).

## Feature List

| Feature      | Cardinality (Approx) | Hashing Strategy          |
| ------------ | -------------------- | ------------------------- |
| `User-Agent` | High (1M+)           | Prefix hash (`ua:{val}`)  |
| `Region`     | Low (100)            | Exact hash (`reg:{val}`)  |
| `City`       | Medium (5k)          | Exact hash (`city:{val}`) |
| `Domain`     | High (500k+)         | Exact hash (`dom:{val}`)  |
| `AdSlot`     | Low (10)             | Exact hash (`fmt:{val}`)  |

_Note: The hashing trick creates a fixed feature space of 262,144 dimensions._

## Performance & Calibration

### Metrics

| Metric          | CTR Model | CVR Model |
| --------------- | --------- | --------- |
| **Log Loss**    | 0.1245    | 0.0412    |
| **AUC-ROC**     | 0.76      | 0.72      |
| **Brier Score** | 0.0031    | 0.0012    |

### Calibration

The model exhibits slight under-confidence in the [0.0, 0.1] probability range.

- **Mitigation**: Post-processing via Platt Scaling is recommended for v2.
- **Safety**: Bidding logic clamps extremely low probabilities to $1e^{-7}$.

## Limitations & Biases

1.  **Cold Start**: New domains/advertisers are hashed to random buckets. Until weights adjust (online learning), predictions are random.
2.  **Hash Collisions**: With $2^{18}$ buckets, collisions are rare but non-zero. A collision between a high-value and low-value feature cancels out the signal.
3.  **Feedback Loop**: The model is trained only on winning bids (selection bias). We use random exploration (epsilon-greedy) to gather counter-factual data.

## Model Integrity

- **Signature**: `SHA256` detached signature required for loading.
- **Drift Detection**: Not currently implemented online. Periodic offline evaluation required.
