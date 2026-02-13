from typing import Dict, List

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calculate_metrics(
    y_true: List[int],
    y_prob: List[float],
    bids: List[float],
    costs: List[float],
    wins: List[bool],
    clicks: List[int],
    conversions: List[int],
    values: List[float],
) -> Dict[str, float]:
    """
    Calculate comprehensive scientific and economic metrics.

    Args:
        y_true: True labels (1=click/conv, 0=no) for model evaluation.
        y_prob: Predicted probabilities.
        bids: Bid amounts placed.
        costs: Actual cost incurred (0 if lost).
        wins: Boolean indicating if bid won.
        clicks: Number of clicks obtained.
        conversions: Number of conversions obtained.
        values: Value generated (revenue).

    Returns:
        Dictionary of metrics.
    """

    # ensure numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bids = np.array(bids)
    costs = np.array(costs)
    wins = np.array(wins)

    # 1. Scientific Metrics (Model Performance)
    # Only calculate if we have variance in y_true (avoid error if all 0 or all 1)
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            ll = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
        else:
            auc = 0.5
            ll = 0.0
            brier = 0.0
    except Exception:
        auc, ll, brier = 0.5, 0.0, 0.0

    # Calibration Error (ECE)
    ece = _expected_calibration_error(y_true, y_prob)

    # 2. Economic Metrics
    total_bids = len(bids)
    total_wins = np.sum(wins)
    total_spend = np.sum(costs)
    total_clicks = np.sum(clicks)
    total_convs = np.sum(conversions)
    total_value = np.sum(values)

    win_rate = total_wins / total_bids if total_bids > 0 else 0.0

    # Efficiency
    cpm = (
        (total_spend / total_wins) * 1000 if total_wins > 0 else 0.0
    )  # Cost per 1000 impressions
    cpc = total_spend / total_clicks if total_clicks > 0 else 0.0
    cpa = total_spend / total_convs if total_convs > 0 else 0.0

    # ROI
    roi = total_value / total_spend if total_spend > 0 else 0.0
    (
        total_value / total_spend if total_spend > 0 else 0.0
    )  # Value is Score in this context?

    return {
        "AUC": auc,
        "LogLoss": ll,
        "Brier": brier,
        "ECE": ece,
        "WinRate": win_rate,
        "TotalSpend": total_spend,
        "TotalValue": total_value,
        "ROI": roi,
        "eCPM": cpm,
        "eCPC": cpc,
        "eCPA": cpa,
        "Clicks": int(total_clicks),
        "Conversions": int(total_convs),
    }


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    """
    if len(y_true) == 0:
        return 0.0

    # Bin predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_prob_in_bin = np.mean(y_prob[in_bin])
            avg_true_in_bin = np.mean(y_true[in_bin])
            ece += np.abs(avg_prob_in_bin - avg_true_in_bin) * prop_in_bin

    return ece
