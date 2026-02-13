# Model Card: Nexus-RTB V1

## Model Details

- **Name**: Nexus-RTB Click/Conv Predictor
- **Version**: 1.0.0
- **Type**: Logistic Regression (Dual model: CTR + CVR)
- **Engine**: Scikit-Learn SGDClassifier
- **License**: MIT

## Intended Use

Real-time prediction of Click-Through Rate (CTR) and Conversion Rate (CVR) for programmatic advertising. Designed for sub-5ms inference latency.

## Training Data

- **Dataset**: IPinYou Global RTB Bidding Algorithm Competition Dataset (Season 2 & 3)
- **Traffic Type**: Display Advertising
- **Volume**: 14M+ impressions from 7 days (2013-06-06 to 2013-06-12)
- **Advertisers**: 5 distinct campaigns (1458, 3358, 3386, 3427, 3476)

## Feature Engineering

Categorical features are encoded using the **hashing trick** (Adler32/SHA256) into a fixed-size vector space.

- **Hash Space**: $2^{18}$ (262,144 features)
- **Features Used**:
  - `ua_os`: Operating System (Windows, Mac, iOS, Android, etc.)
  - `ua_browser`: Browser (Chrome, Safari, Firefox, etc.)
  - `region`: Geographic region ID
  - `city`: City ID
  - `adslot_visibility`: FirstView, SecondView, etc.
  - `adslot_format`: Fixed, Popup, etc.
  - `advertiser`: Advertiser ID
  - `domain`: Publisher domain

## Performance

- **Training Method**: Streaming SGD (Stochastic Gradient Descent) with `log_loss`
- **Validation**: Online evaluation on Day 08-12 logs
- **Inference Latency**: < 0.05ms (core model dot product), < 2ms (end-to-end extraction)

## Limitations & Bias

- **Data Freshness**: Trained on 2013 data; user agent patterns and browsing behaviors are likely outdated.
- **Geography**: Heavily skewed towards Chinese traffic (IPinYou dataset).
- **Cold Start**: New domains or user agents hash to buckets without history (collisions may add noise).
- **Bias**: No fairness constraints were applied. The model optimizes purely for click/conversion probability.

## Fail-Safe Mechanisms

- If model weights fail to load, the engine defaults to conservative intercepts (~1.8% probability) to prevent overbidding.
