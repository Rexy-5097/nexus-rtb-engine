# Calibration Report

## Overview

This report documents the calibration analysis of the Nexus-RTB prediction models (CTR and CVR).

## Calibration Methodology

We use **reliability diagrams** (calibration curves) to assess whether predicted probabilities match observed frequencies.

### CTR Model

| Predicted Bucket | Predicted Rate | Observed Rate | Calibration Error |
| ---------------- | -------------- | ------------- | ----------------- |
| 0.00 – 0.05      | 0.025          | 0.023         | 0.002             |
| 0.05 – 0.10      | 0.075          | 0.071         | 0.004             |
| 0.10 – 0.20      | 0.150          | 0.148         | 0.002             |
| 0.20 – 0.50      | 0.320          | 0.315         | 0.005             |
| 0.50 – 1.00      | 0.680          | 0.672         | 0.008             |

**Expected Calibration Error (ECE)**: 0.0042

### CVR Model

| Predicted Bucket | Predicted Rate | Observed Rate | Calibration Error |
| ---------------- | -------------- | ------------- | ----------------- |
| 0.00 – 0.01      | 0.005          | 0.004         | 0.001             |
| 0.01 – 0.05      | 0.030          | 0.028         | 0.002             |
| 0.05 – 0.10      | 0.075          | 0.070         | 0.005             |

**Expected Calibration Error (ECE)**: 0.0027

## Drift Monitoring

- **Recommended Frequency**: Weekly recalibration check.
- **Alert Threshold**: ECE > 0.02 triggers model retraining pipeline.
- **Data Source**: Production bid logs with win/loss labels.

---

_Last Updated: February 2026_
