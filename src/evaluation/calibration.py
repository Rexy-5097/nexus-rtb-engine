import logging
from typing import List

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class Calibrator:
    """
    Evaluates the calibration of the CTR and CVR models.
    """

    @staticmethod
    def evaluate(y_true: List[int], y_prob: List[float], bins=10):
        """
        Compute calibration metrics and reliability diagram data.
        """
        if not y_true:
            return {}

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # Metrics
        ll = log_loss(y_true, y_prob)
        bs = brier_score_loss(y_true, y_prob)

        # Reliability Diagram (Binning)
        # Bin predictions into 10 buckets (0-0.1, 0.1-0.2, ...)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1

        prob_true = []
        prob_pred = []
        bin_counts = []

        for i in range(bins):
            mask = bin_indices == i
            if np.any(mask):
                prob_true.append(y_true[mask].mean())
                prob_pred.append(y_prob[mask].mean())
                bin_counts.append(int(mask.sum()))
            else:
                prob_true.append(0.0)
                prob_pred.append(bin_edges[i] + 0.05)  # Center of empty bin
                bin_counts.append(0)

        return {
            "log_loss": ll,
            "brier_score": bs,
            "reliability_diagram": {
                "prob_true": prob_true,
                "prob_pred": prob_pred,
                "counts": bin_counts,
            },
        }

    @staticmethod
    def generate_ascii_plot(reliability_data):
        """Generate a simple ASCII Reliability Diagram."""
        pred = reliability_data["prob_pred"]
        true = reliability_data["prob_true"]
        counts = reliability_data["counts"]

        lines = []
        lines.append("\n=== Reliability Diagram ===")
        lines.append("Pred   | Obs    | Count | Bar")
        lines.append("-------|--------|-------|----")

        for p, t, c in zip(pred, true, counts):
            if c == 0:
                continue
            # Visualize gap
            t - p
            bar_len = int(t * 50)
            bar = "#" * bar_len
            lines.append(f"{p:.3f}  | {t:.3f}  | {c:5d} | {bar}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    y_true = [0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    y_prob = [0.1, 0.2, 0.1, 0.8, 0.3, 0.9, 0.7, 0.8, 0.4, 0.2]
    res = Calibrator.evaluate(y_true, y_prob, bins=5)
    print(f"Log Loss: {res['log_loss']}")
    print(Calibrator.generate_ascii_plot(res["reliability_diagram"]))
