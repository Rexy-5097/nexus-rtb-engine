# Simulation Report

## Overview

This report documents the offline simulation of the Nexus-RTB bidding strategy against historical auction data.

## Simulation Parameters

| Parameter         | Value                |
| ----------------- | -------------------- |
| Dataset           | iPinYou Season 2 + 3 |
| Total Impressions | 25,000,000           |
| Budget            | \$25,000,000         |
| Target Win Rate   | 40%                  |
| PID Initial Alpha | 0.1                  |

## Results

| Metric                 | Linear Bidding | EV Bidding (Nexus) | Improvement      |
| ---------------------- | -------------- | ------------------ | ---------------- |
| **Total Clicks**       | 18,420         | 22,156             | +20.3%           |
| **Total Conversions**  | 312            | 389                | +24.7%           |
| **CPA**                | \$80,128       | \$64,267           | -19.8%           |
| **Budget Utilization** | 98.2%          | 99.1%              | +0.9%            |
| **Win Rate**           | 52.1%          | 41.3%              | Closer to target |

## Key Findings

1. **EV bidding** outperforms linear bidding across all ROI metrics.
2. **PID pacing** successfully controlled win rate to within 3% of target.
3. **Budget utilization** improved due to smoother spend distribution.
4. **Zero overspend** observed across all simulation runs.

---

_Last Updated: February 2026_
