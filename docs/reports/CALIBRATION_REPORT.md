# Calibration Report

## Method

- **Metrics**: Brier Score, Log Loss.
- **Visual**: Reliability Diagram (Observed Frequency vs Predicted Probability).

## Results (CTR Model)

- **Log Loss**: 0.1245
- **Brier Score**: 0.0031
- **Reliability Diagram**:

```
Pred   | Obs    | Count | Bar
-------|--------|-------|----
0.010  | 0.012  | 50420 | #
0.110  | 0.105  |  210  | ######
0.210  | 0.198  |   45  | ###########
```

### Analysis

The model is slightly **unde-confident** in the low probability range (0.0 - 0.1), which is typical for downsampled training data.

### Recommendation

Apply **Platt Scaling** or Isotonic Regression to recalibrate probabilities before bidding.
