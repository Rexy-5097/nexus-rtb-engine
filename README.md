# Nexus-RTB: High-Frequency Request-to-Bid Engine

**Nexus-RTB** is a production-grade, high-frequency Real-Time Bidding (RTB) engine designed for second-price auctions. It is optimized for extreme low-latency environments (< 5ms per inference) and constrained memory usage, making it suitable for edge deployment or high-throughput DSPs.

## 🚀 Key Features

- **Ultra-Low Latency**: Pure Python implementation using hashed feature lookups and lightweight logistic regression, ensuring <5ms response times.
- **Market Anchoring**: Bidding strategy is anchored to historical market price (2nd price) rather than pure probability, optimizing for win-rate in second-price auctions.
- **Adaptive Pacing**: Implements a feedback-loop control system (PID-like) to smooth budget consumption throughout the day, preventing early budget exhaustion.
- **Hybrid Prediction**: Uses dual Logistic Regression models (CTR + CVR) trained with SGD on 14M+ impressions.
- **Safety Gates**: Built-in dynamic floor price compliance and Minimum Expected Value (EV) gating to reject low-quality inventory.

## 📂 Project Structure

```
nexus-rtb-engine/
├── src/
│   ├── Bid.py            # Core bidding logic (Inference Engine)
│   ├── Bidder.py         # Interface definition
│   ├── BidRequest.py     # Request object model
│   └── model_weights.pkl # Serialized model weights & stats
├── training/
│   ├── train.py          # Streaming training pipeline (SGD)
│   └── debug_data.py     # Data inspection utilities
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## 🛠️ Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Rexy-5097/nexus-rtb-engine.git
    cd nexus-rtb-engine
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Training (Optional):**
    If you have the dataset (IPinYou or compatible), you can retrain the model:

    ```bash
    python training/train.py
    ```

    This will generate a new `model_weights.pkl`.

4.  **Inference:**
    The `Bid.py` class is designed to be instantiated by a DSP framework.

    ```python
    from src.Bid import Bid
    from src.BidRequest import BidRequest

    bidder = Bid()
    # ... receive request_data ...
    price = bidder.getBidPrice(request_data)
    ```

## 🧠 Algorithmic Approach

### Feature Engineering

High-cardinality categorical features (User-Agent, City, Domain, etc.) are hashed into a fixed-size space ($2^{18}$) using the `hashing trick`. This ensures bounded memory usage (< 512MB) and constant-time lookup.

### Bidding Logic

The bid price is calculated as:
$$ Bid = MarketPrice*{avg} \times \min(3.0, \frac{EV}{EV*{avg}}) \times PacingFactor $$
Where $EV = pCTR + N \times pCTR \times pCVR$.

This formula ensures we bid competitively for high-value users while respecting the advertiser's specific conversion goals ($N$).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
