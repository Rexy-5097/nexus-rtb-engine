#!/usr/bin/env python3
"""
Phase 11: Real-Time Market & Signal Realism
=============================================
1. Business Metric Alignment (DR → RPM / Profit)
2. Pareto Frontier for Multi-Objective Optimization
3. Adaptive EV Gate Scheduler (PID, 48h)
4. Delayed Feedback Modeling
5. Game-Theoretic Competitor Reaction (10-round)
"""
import logging
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

from scipy.sparse import vstack  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from src.bidding.config import config  # noqa: E402
from src.training.train import (FeatureExtractor, build_matrix,  # noqa: E402
                                load_dataset)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# ──────────────────── HELPERS ────────────────────
def prepare_data():
    full_df, global_stats, scaler, stats, top_k_maps = load_dataset()
    fe = FeatureExtractor()
    fe.set_encoding_maps(top_k_maps)
    X_parts, y_ctr_parts, y_cvr_parts = [], [], []
    for start in range(0, len(full_df), 100000):
        chunk = full_df.iloc[start : start + 100000]
        c = chunk["click"].values.astype(np.int8)
        v = chunk["conversion"].values.astype(np.int8)
        rows = list(chunk.itertuples(index=False))
        mat = build_matrix(rows, c, v, global_stats, fe, scaler)
        if mat is not None:
            X_parts.append(mat)
            y_ctr_parts.append(c)
            y_cvr_parts.append(v)
    X = vstack(X_parts)
    y_ctr = np.concatenate(y_ctr_parts)
    y_cvr = np.concatenate(y_cvr_parts)
    prices = full_df["payingprice"].values.astype(float)
    return X, y_ctr, y_cvr, prices


def auc_safe(y, p):
    try:
        return roc_auc_score(y, p)
    except:
        return 0.5


BASE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 10.0,
    "lambda_l2": 10.0,
}


def train_lgbm(X_tr, y_tr, X_va, y_va, params, spw=1.0):
    p = dict(
        objective="binary", metric="auc", verbose=-1, scale_pos_weight=spw, **params
    )
    m = lgb.LGBMClassifier(**p)
    m.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return m


def calc_roi(clicks, convs, spend):
    value = clicks * config.value_click + convs * config.value_conversion
    return value / max(spend, 1)


def run_backtest(pCTR, prices, y_click, y_conv, ev_pct=70, budget_frac=0.8):
    ev = pCTR * config.value_click
    BUDGET = float(prices.sum()) * budget_frac
    ev_thresh = np.percentile(ev, ev_pct) if ev_pct > 0 else 0
    spend, wins, clicks, convs = 0.0, 0, 0, 0
    for i in range(len(prices)):
        mp = prices[i]
        if ev_pct > 0 and ev[i] < ev_thresh:
            continue
        if spend + mp <= BUDGET and ev[i] >= mp:
            spend += mp
            wins += 1
            if y_click[i]:
                clicks += 1
            if y_conv[i]:
                convs += 1
    return {
        "roi": calc_roi(clicks, convs, spend),
        "spend": spend,
        "budget": BUDGET,
        "wins": wins,
        "clicks": clicks,
        "convs": convs,
        "util": spend / BUDGET,
        "win_rate": wins / max(len(prices), 1),
    }


# ═══════════════════════════════════════════════════════════════
# 1. BUSINESS METRIC ALIGNMENT
# ═══════════════════════════════════════════════════════════════
def task1_business_metrics(X, y_ctr, y_cvr, prices):
    logger.info("=" * 65)
    logger.info("TASK 1: BUSINESS METRIC ALIGNMENT (DR → Economic KPIs)")
    logger.info("=" * 65)

    n = X.shape[0]
    sp = int(n * 0.7)
    X_tr, y_tr = X[:sp], y_ctr[:sp]
    X_te, y_te = X[sp:], y_ctr[sp:]
    y_click_te, y_conv_te, prices_te = y_ctr[sp:], y_cvr[sp:], prices[sp:]
    spw = (1.0 - y_tr.mean()) / (y_tr.mean() + 1e-6)

    # Logging policy (weaker)
    log_p = dict(BASE_PARAMS, n_estimators=100, num_leaves=8)
    log_model = train_lgbm(X_tr, y_tr, X_te, y_te, log_p, spw)
    p_log = np.clip(log_model.predict_proba(X_te)[:, 1], 0.01, 0.99)

    # Target policy (production)
    tgt_model = train_lgbm(X_tr, y_tr, X_te, y_te, BASE_PARAMS, spw)
    p_tgt = np.clip(tgt_model.predict_proba(X_te)[:, 1], 0.01, 0.99)

    # Reward model
    rwd_model = train_lgbm(
        X_tr, y_tr, X_te, y_te, dict(BASE_PARAMS, n_estimators=100), spw
    )
    mu_hat = rwd_model.predict_proba(X_te)[:, 1]

    threshold = np.median(p_log)
    logged_action = (p_log >= threshold).astype(int)
    propensity = np.clip(np.where(logged_action == 1, p_log, 1 - p_log), 0.05, 0.95)
    target_prob = np.where(logged_action == 1, p_tgt, 1 - p_tgt)
    w = np.clip(target_prob / propensity, 0.1, 10.0)
    rewards = y_te.copy().astype(float)
    rewards[logged_action == 0] = 0

    # DR estimate
    dr_direct = np.mean(mu_hat * target_prob)
    dr_correction = np.mean(w * (rewards - mu_hat * logged_action))
    dr_estimate = dr_direct + dr_correction

    # Replay ROI
    bt = run_backtest(p_tgt, prices_te, y_click_te, y_conv_te, ev_pct=70)

    # Map to economic units
    n_impressions = len(prices_te)
    total_value = (
        bt["clicks"] * config.value_click + bt["convs"] * config.value_conversion
    )

    rpm_replay = (total_value / n_impressions) * 1000  # Revenue per mille
    rpm_dr = dr_estimate * config.value_click * 1000  # DR-based RPM

    profit_per_1k_spend = ((total_value - bt["spend"]) / max(bt["spend"], 1)) * 1000
    dr_profit_per_1k = (
        (dr_estimate * config.value_click * n_impressions - bt["spend"])
        / max(bt["spend"], 1)
    ) * 1000

    uplift_pct = (rpm_dr - rpm_replay) / max(rpm_replay, 0.01) * 100

    logger.info(f"\n  {'Metric':<35s} {'Replay':<15s} {'DR-Based':<15s} {'Unit'}")
    logger.info(f"  {'─'*75}")
    logger.info(
        f"  {'RPM (Revenue per 1K imps)':<35s} {rpm_replay:<15.2f} {rpm_dr:<15.2f} $/1000 imps"
    )
    logger.info(
        f"  {'Profit per $1000 spend':<35s} {profit_per_1k_spend:<15.2f} {dr_profit_per_1k:<15.2f} $/1000 spend"
    )
    logger.info(f"  {'ROI':<35s} {bt['roi']:<15.4f} {'N/A':<15s} ratio")
    logger.info(
        f"  {'DR Policy Value':<35s} {'N/A':<15s} {dr_estimate:<15.6f} prob-units"
    )
    logger.info(f"  {'Economic Uplift':<35s} {uplift_pct:+.1f}%")
    logger.info(f"  {'Impressions':<35s} {n_impressions}")
    logger.info(f"  {'Total Value (Replay)':<35s} {total_value:.0f}")

    return {
        "rpm_replay": round(rpm_replay, 2),
        "rpm_dr": round(rpm_dr, 2),
        "profit_per_1k_replay": round(profit_per_1k_spend, 2),
        "profit_per_1k_dr": round(dr_profit_per_1k, 2),
        "roi_replay": round(bt["roi"], 4),
        "dr_estimate": round(dr_estimate, 6),
        "uplift_pct": round(uplift_pct, 1),
    }


# ═══════════════════════════════════════════════════════════════
# 2. PARETO FRONTIER
# ═══════════════════════════════════════════════════════════════
def task2_pareto(X, y_ctr, y_cvr, prices):
    logger.info("=" * 65)
    logger.info("TASK 2: PARETO FRONTIER (50-point Multi-Objective)")
    logger.info("=" * 65)

    n = X.shape[0]
    sp = int(n * 0.7)
    X_tr, y_tr = X[:sp], y_ctr[:sp]
    X_te, y_te = X[sp:], y_ctr[sp:]
    y_click_te, y_conv_te, prices_te = y_ctr[sp:], y_cvr[sp:], prices[sp:]
    spw = (1.0 - y_tr.mean()) / (y_tr.mean() + 1e-6)

    model = train_lgbm(X_tr, y_tr, X_te, y_te, BASE_PARAMS, spw)
    pCTR = model.predict_proba(X_te)[:, 1]
    ev = pCTR * config.value_click
    BUDGET = float(prices_te.sum()) * 0.8
    n_te = len(prices_te)

    # Sample 50 Lagrangian configs: sweep λ_cpa and λ_util
    configs = []
    for lc in np.linspace(0, 5, 10):  # λ_cpa
        for lu in np.linspace(0, 0.5, 5):  # λ_util
            configs.append((lc, lu))

    results = []
    logger.info(
        f"\n  {'#':<4} {'λ_CPA':<8} {'λ_Util':<8} {'ROI':<8} {'Util':<8} {'CPA':<10} {'WinRate':<10} {'Profit':<10}"
    )

    for idx, (lc, lu) in enumerate(configs):
        bid_adj = ev * (1 + lu) / (1 + lc) if (1 + lc) > 0 else ev
        spend, wins, clicks, convs = 0.0, 0, 0, 0
        for i in range(n_te):
            mp = prices_te[i]
            if spend + mp <= BUDGET and bid_adj[i] >= mp:
                spend += mp
                wins += 1
                if y_click_te[i]:
                    clicks += 1
                if y_conv_te[i]:
                    convs += 1
        roi = calc_roi(clicks, convs, spend)
        util = spend / BUDGET
        cpa = spend / max(convs, 1)
        profit = clicks * config.value_click + convs * config.value_conversion - spend
        wr = wins / n_te

        results.append(
            {
                "idx": idx,
                "lambda_cpa": round(lc, 2),
                "lambda_util": round(lu, 2),
                "roi": round(roi, 4),
                "util": round(util, 4),
                "cpa": round(cpa, 1),
                "win_rate": round(wr, 4),
                "profit": round(profit, 1),
                "wins": wins,
                "convs": convs,
            }
        )

        if idx % 10 == 0 or idx == len(configs) - 1:
            logger.info(
                f"  {idx:<4} {lc:<8.2f} {lu:<8.2f} {roi:<8.4f} {util:<8.4f} "
                f"{cpa:<10.1f} {wr:<10.4f} {profit:<10.1f}"
            )

    # Identify Pareto-efficient points (ROI vs Util)
    pareto = []
    for r in sorted(results, key=lambda x: x["util"]):
        if not pareto or r["roi"] > pareto[-1]["roi"]:
            pareto.append(r)

    logger.info(f"\n  Pareto-Efficient Frontier ({len(pareto)} points):")
    logger.info(
        f"  {'Util':<10} {'ROI':<10} {'CPA':<10} {'WinRate':<10} {'Profit':<10}"
    )
    for p in pareto:
        logger.info(
            f"  {p['util']:<10.4f} {p['roi']:<10.4f} {p['cpa']:<10.1f} "
            f"{p['win_rate']:<10.4f} {p['profit']:<10.1f}"
        )

    # Recommended operating region: highest profit on Pareto
    best = max(pareto, key=lambda x: x["profit"])
    logger.info(
        f"\n  📈 Recommended operating point: Util={best['util']:.1%}, ROI={best['roi']:.4f}, "
        f"Profit={best['profit']:.0f}"
    )

    return {"all_points": results, "pareto": pareto, "recommended": best}


# ═══════════════════════════════════════════════════════════════
# 3. ADAPTIVE EV GATE SCHEDULER (PID, 48h)
# ═══════════════════════════════════════════════════════════════
def task3_adaptive_gate(X, y_ctr, y_cvr, prices):
    logger.info("=" * 65)
    logger.info("TASK 3: ADAPTIVE EV GATE SCHEDULER (PID, 48h)")
    logger.info("=" * 65)

    n = X.shape[0]
    sp = int(n * 0.7)
    X_tr, y_tr = X[:sp], y_ctr[:sp]
    X_te, y_te = X[sp:], y_ctr[sp:]
    y_click_te, y_conv_te, prices_te = y_ctr[sp:], y_cvr[sp:], prices[sp:]
    spw = (1.0 - y_tr.mean()) / (y_tr.mean() + 1e-6)

    model = train_lgbm(X_tr, y_tr, X_te, y_te, BASE_PARAMS, spw)
    pCTR = model.predict_proba(X_te)[:, 1]
    ev = pCTR * config.value_click
    n_te = len(prices_te)

    # 48h = replicate test data twice
    ev_48 = np.tile(ev, 2)[: n_te * 2]
    prices_48 = np.tile(prices_te, 2)[: n_te * 2]
    y_click_48 = np.tile(y_click_te, 2)[: n_te * 2]
    y_conv_48 = np.tile(y_conv_te, 2)[: n_te * 2]
    total_n = len(ev_48)

    HOURS = 48
    imps_per_hour = total_n // HOURS
    BUDGET = float(prices_48.sum()) * 0.8
    BUDGET / HOURS

    # PID controller for EV percentile gate
    TARGET_UTIL = 0.80
    TARGET_ROI = 0.85
    gate_pct = 70.0  # Starting gate percentile
    Kp_util = 50.0  # Proportional gain (utilization)
    Kp_roi = 30.0  # Proportional gain (ROI)
    Ki_util = 5.0  # Integral gain
    integral_err = 0.0

    cum_spend, cum_clicks, cum_convs, cum_wins = 0.0, 0, 0, 0
    hourly_results = []

    logger.info(
        f"  Budget: {BUDGET:.0f}, Target Util: {TARGET_UTIL:.0%}, Target ROI: {TARGET_ROI}"
    )
    logger.info(
        f"\n  {'Hour':<6} {'Gate%':<8} {'Spend':<10} {'CumUtil':<10} {'ROI':<8} "
        f"{'Wins':<6} {'Clicks':<8}"
    )

    for hour in range(HOURS):
        h_start = hour * imps_per_hour
        h_end = min(h_start + imps_per_hour, total_n)

        # Current gate threshold
        gate_thresh = np.percentile(ev_48, gate_pct) if gate_pct > 0 else 0
        hour_spend, hour_wins, hour_clicks, hour_convs = 0.0, 0, 0, 0

        for i in range(h_start, h_end):
            mp = prices_48[i]
            if ev_48[i] >= gate_thresh and cum_spend + mp <= BUDGET and ev_48[i] >= mp:
                cum_spend += mp
                hour_spend += mp
                hour_wins += 1
                cum_wins += 1
                if y_click_48[i]:
                    hour_clicks += 1
                    cum_clicks += 1
                if y_conv_48[i]:
                    hour_convs += 1
                    cum_convs += 1

        # Compute error signals
        elapsed_frac = (hour + 1) / HOURS
        BUDGET * TARGET_UTIL * elapsed_frac
        actual_util = cum_spend / BUDGET
        expected_util = TARGET_UTIL * elapsed_frac
        util_error = expected_util - actual_util  # Positive = underspending

        current_roi = calc_roi(cum_clicks, cum_convs, cum_spend)
        roi_error = current_roi - TARGET_ROI  # Positive = ROI too high (can relax)

        # PID update
        integral_err += util_error * 0.1
        integral_err = np.clip(integral_err, -5, 5)

        # If underspending → lower gate; if ROI too low → raise gate
        gate_adjustment = (
            -Kp_util * util_error
            + Kp_roi * max(0, -roi_error)
            + Ki_util * (-integral_err)
        )
        gate_pct = np.clip(gate_pct + gate_adjustment, 0, 95)

        hourly_results.append(
            {
                "hour": hour,
                "gate_pct": round(gate_pct, 1),
                "spend": round(hour_spend, 0),
                "cum_util": round(actual_util, 4),
                "roi": round(current_roi, 4),
                "wins": hour_wins,
                "clicks": hour_clicks,
            }
        )

        if hour % 6 == 0 or hour == HOURS - 1:
            logger.info(
                f"  H{hour:<5} {gate_pct:<8.1f} {hour_spend:<10.0f} {actual_util:<10.4f} "
                f"{current_roi:<8.4f} {hour_wins:<6} {hour_clicks:<8}"
            )

    final_util = cum_spend / BUDGET
    final_roi = calc_roi(cum_clicks, cum_convs, cum_spend)
    logger.info(
        f"\n  48h Summary: Util={final_util:.1%}, ROI={final_roi:.4f}, "
        f"Wins={cum_wins}, Clicks={cum_clicks}, Convs={cum_convs}"
    )
    logger.info(
        f"  Gate range: {min(h['gate_pct'] for h in hourly_results):.0f}% → "
        f"{max(h['gate_pct'] for h in hourly_results):.0f}%"
    )

    return {
        "hourly": hourly_results,
        "final_util": round(final_util, 4),
        "final_roi": round(final_roi, 4),
        "total_wins": cum_wins,
        "gate_range": [
            min(h["gate_pct"] for h in hourly_results),
            max(h["gate_pct"] for h in hourly_results),
        ],
    }


# ═══════════════════════════════════════════════════════════════
# 4. DELAYED FEEDBACK MODELING
# ═══════════════════════════════════════════════════════════════
def task4_delayed_feedback(X, y_ctr, y_cvr, prices):
    logger.info("=" * 65)
    logger.info("TASK 4: DELAYED FEEDBACK MODELING")
    logger.info("=" * 65)

    n = X.shape[0]
    sp = int(n * 0.7)
    y_click_te, y_conv_te, prices_te = y_ctr[sp:], y_cvr[sp:], prices[sp:]
    n_te = len(prices_te)

    # Simulate delay distributions
    np.random.seed(42)
    click_delays = np.random.exponential(30, n_te)  # Mean 30 min
    conv_delays = np.random.exponential(240, n_te)  # Mean 4 hours (240 min)
    observation_window = 60  # 60-minute observation window

    # Observed feedback (within observation window)
    click_observed = y_click_te.copy().astype(float)
    conv_observed = y_conv_te.copy().astype(float)

    # Mark delayed feedback as missing (not yet observed)
    click_delayed_mask = click_delays > observation_window
    conv_delayed_mask = conv_delays > observation_window

    click_observed[click_delayed_mask & (y_click_te == 1)] = 0  # Missed clicks
    conv_observed[conv_delayed_mask & (y_conv_te == 1)] = 0  # Missed conversions

    # Naive estimator (uses only observed feedback)
    naive_ctr = click_observed.sum() / n_te
    naive_cvr = conv_observed[click_observed == 1].sum() / max(click_observed.sum(), 1)
    true_ctr = y_click_te.sum() / n_te
    true_cvr = y_conv_te[y_click_te == 1].sum() / max(y_click_te.sum(), 1)

    # Delay-aware correction using importance weighting
    # P(observed | event) = P(delay < window) = 1 - exp(-window/mean_delay)
    p_click_obs = 1 - np.exp(-observation_window / 30)  # ~86.5%
    p_conv_obs = 1 - np.exp(-observation_window / 240)  # ~22.1%

    corrected_ctr = click_observed.sum() / (n_te * p_click_obs)
    corrected_clicks = click_observed.sum()
    corrected_cvr_denom = corrected_clicks / p_click_obs
    corrected_cvr = conv_observed[click_observed == 1].sum() / (
        max(corrected_cvr_denom, 1) * p_conv_obs
    )

    # Bayesian posterior update
    # Prior: Beta(alpha=1, beta=1) (uninformative)
    # Posterior: Beta(alpha + observed, beta + N - observed)
    alpha_prior, beta_prior = 1.0, 1.0
    alpha_post = alpha_prior + click_observed.sum()
    beta_post = beta_prior + n_te - click_observed.sum()
    bayesian_ctr = alpha_post / (alpha_post + beta_post)

    # Corrected Bayesian (adjust N for observation probability)
    effective_n = n_te * p_click_obs
    alpha_corrected = alpha_prior + click_observed.sum()
    beta_corrected = beta_prior + effective_n - click_observed.sum()
    bayesian_corrected_ctr = alpha_corrected / (alpha_corrected + beta_corrected)

    # Lag-aware ROI estimator
    naive_value = (
        click_observed.sum() * config.value_click
        + conv_observed.sum() * config.value_conversion
    )
    corrected_value = (click_observed.sum() / p_click_obs) * config.value_click + (
        conv_observed.sum() / p_conv_obs
    ) * config.value_conversion
    true_value = (
        y_click_te.sum() * config.value_click
        + y_conv_te.sum() * config.value_conversion
    )
    total_spend = prices_te.sum() * 0.3  # Assume 30% win rate spend

    naive_roi = naive_value / max(total_spend, 1)
    corrected_roi = corrected_value / max(total_spend, 1)
    true_roi = true_value / max(total_spend, 1)

    # Bias analysis
    ctr_bias_naive = (naive_ctr - true_ctr) / max(true_ctr, 1e-6) * 100
    ctr_bias_corrected = (corrected_ctr - true_ctr) / max(true_ctr, 1e-6) * 100
    roi_bias_naive = (naive_roi - true_roi) / max(true_roi, 1e-6) * 100
    roi_bias_corrected = (corrected_roi - true_roi) / max(true_roi, 1e-6) * 100

    logger.info("\n  Delay Parameters:")
    logger.info(f"    Click delay: Exp(μ=30min), P(observe)={p_click_obs:.3f}")
    logger.info(f"    Conv delay:  Exp(μ=4h),    P(observe)={p_conv_obs:.3f}")
    logger.info(f"    Observation window: {observation_window} min")

    logger.info(
        f"\n  {'Metric':<30s} {'True':<12s} {'Naive':<12s} {'Corrected':<12s} {'Bias(Naive)':<14s} {'Bias(Corr)'}"
    )
    logger.info(f"  {'─'*85}")
    logger.info(
        f"  {'CTR':<30s} {true_ctr:<12.6f} {naive_ctr:<12.6f} {corrected_ctr:<12.6f} "
        f"{ctr_bias_naive:<+14.1f}% {ctr_bias_corrected:<+.1f}%"
    )
    logger.info(
        f"  {'CVR':<30s} {true_cvr:<12.6f} {naive_cvr:<12.6f} {corrected_cvr:<12.6f} "
        f"{'—':<14s} {'—'}"
    )
    logger.info(
        f"  {'ROI':<30s} {true_roi:<12.4f} {naive_roi:<12.4f} {corrected_roi:<12.4f} "
        f"{roi_bias_naive:<+14.1f}% {roi_bias_corrected:<+.1f}%"
    )
    logger.info(
        f"  {'Bayesian CTR':<30s} {'—':<12s} {bayesian_ctr:<12.6f} {bayesian_corrected_ctr:<12.6f}"
    )

    n_missed_clicks = int((click_delayed_mask & (y_click_te == 1)).sum())
    n_missed_convs = int((conv_delayed_mask & (y_conv_te == 1)).sum())
    logger.info(
        f"\n  Missed clicks: {n_missed_clicks}/{int(y_click_te.sum())} "
        f"({n_missed_clicks/max(y_click_te.sum(),1)*100:.1f}%)"
    )
    logger.info(
        f"  Missed conversions: {n_missed_convs}/{int(y_conv_te.sum())} "
        f"({n_missed_convs/max(y_conv_te.sum(),1)*100:.1f}%)"
    )

    return {
        "true_ctr": round(true_ctr, 6),
        "naive_ctr": round(naive_ctr, 6),
        "corrected_ctr": round(corrected_ctr, 6),
        "ctr_bias_naive": round(ctr_bias_naive, 1),
        "ctr_bias_corrected": round(ctr_bias_corrected, 1),
        "true_roi": round(true_roi, 4),
        "naive_roi": round(naive_roi, 4),
        "corrected_roi": round(corrected_roi, 4),
        "roi_bias_naive": round(roi_bias_naive, 1),
        "roi_bias_corrected": round(roi_bias_corrected, 1),
        "p_click_obs": round(p_click_obs, 4),
        "p_conv_obs": round(p_conv_obs, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 5. GAME-THEORETIC COMPETITOR REACTION (10-round)
# ═══════════════════════════════════════════════════════════════
def task5_game_theory(X, y_ctr, y_cvr, prices):
    logger.info("=" * 65)
    logger.info("TASK 5: GAME-THEORETIC COMPETITOR REACTION (10-round)")
    logger.info("=" * 65)

    n = X.shape[0]
    sp = int(n * 0.7)
    X_tr, y_tr = X[:sp], y_ctr[:sp]
    X_te, y_te = X[sp:], y_ctr[sp:]
    y_click_te, y_conv_te, prices_te = y_ctr[sp:], y_cvr[sp:], prices[sp:]
    spw = (1.0 - y_tr.mean()) / (y_tr.mean() + 1e-6)

    model = train_lgbm(X_tr, y_tr, X_te, y_te, BASE_PARAMS, spw)
    pCTR = model.predict_proba(X_te)[:, 1]
    ev = pCTR * config.value_click
    n_te = len(prices_te)
    BUDGET = float(prices_te.sum()) * 0.8

    # Initial competitor distribution
    mu_comp = np.log(prices_te.mean())
    sigma_comp = 0.5
    comp_multiplier = 1.0

    # Our bid shade factor
    our_shade = 0.85

    rounds = []
    logger.info(
        f"\n  {'Round':<7} {'CompMult':<10} {'OurShade':<10} {'OurWins':<10} {'OurROI':<10} "
        f"{'CompWins':<10} {'CompROI':<10} {'Equilibrium'}"
    )

    for rnd in range(10):
        # Generate competitor bids for this round
        comp_bids = np.random.lognormal(mu_comp, sigma_comp, n_te) * comp_multiplier

        # Our bids
        our_bids = ev * our_shade

        our_spend, our_wins, our_clicks, our_convs = 0.0, 0, 0, 0
        comp_spend, comp_wins, comp_clicks, comp_convs = 0.0, 0, 0, 0

        for i in range(n_te):
            mp = prices_te[i]
            our_bid = our_bids[i]
            comp_bid = comp_bids[i]

            if our_bid >= comp_bid and our_bid >= mp:
                # We win
                if our_spend + mp <= BUDGET:
                    our_spend += mp
                    our_wins += 1
                    if y_click_te[i]:
                        our_clicks += 1
                    if y_conv_te[i]:
                        our_convs += 1
            elif comp_bid >= mp:
                # Competitor wins
                comp_spend += mp
                comp_wins += 1
                if y_click_te[i]:
                    comp_clicks += 1
                if y_conv_te[i]:
                    comp_convs += 1

        our_roi = calc_roi(our_clicks, our_convs, our_spend)
        comp_roi = calc_roi(comp_clicks, comp_convs, comp_spend)
        our_wr = our_wins / n_te
        comp_wr = comp_wins / n_te

        # Determine equilibrium status
        eq_status = ""
        if abs(our_wr - comp_wr) < 0.05:
            eq_status = "⚖️ Near equilibrium"
        elif our_wr > comp_wr + 0.1:
            eq_status = "📈 We dominate"
        else:
            eq_status = "📉 Competitor leads"

        rounds.append(
            {
                "round": rnd + 1,
                "comp_mult": round(comp_multiplier, 3),
                "our_shade": round(our_shade, 3),
                "our_wins": our_wins,
                "our_roi": round(our_roi, 4),
                "our_wr": round(our_wr, 4),
                "comp_wins": comp_wins,
                "comp_roi": round(comp_roi, 4),
                "comp_wr": round(comp_wr, 4),
                "eq_status": eq_status,
            }
        )

        logger.info(
            f"  R{rnd+1:<6} {comp_multiplier:<10.3f} {our_shade:<10.3f} {our_wins:<10} "
            f"{our_roi:<10.4f} {comp_wins:<10} {comp_roi:<10.4f} {eq_status}"
        )

        # ── Competitor adaptation ──
        # If losing too often → increase bids
        if comp_wr < 0.3:
            comp_multiplier *= 1.15  # 15% more aggressive
        elif comp_roi < 0.5:
            comp_multiplier *= 0.90  # Pull back if ROI negative
        else:
            comp_multiplier *= 1.02  # Slow escalation

        # ── Our adaptation ──
        # If ROI is strong → can be more aggressive
        if our_roi > 1.5 and our_wr < 0.35:
            our_shade = min(1.0, our_shade + 0.03)
        elif our_roi < 0.8:
            our_shade = max(0.5, our_shade - 0.05)

    # Equilibrium analysis
    final = rounds[-1]
    logger.info("\n  Final State (Round 10):")
    logger.info(f"    Our WinRate: {final['our_wr']:.1%}, ROI: {final['our_roi']:.4f}")
    logger.info(
        f"    Comp WinRate: {final['comp_wr']:.1%}, ROI: {final['comp_roi']:.4f}"
    )
    logger.info(f"    Comp Multiplier: {final['comp_mult']:.3f}× (started at 1.0×)")
    logger.info(f"    Our Shade: {final['our_shade']:.3f} (started at 0.85)")

    # ROI stability
    our_rois = [r["our_roi"] for r in rounds]
    roi_std = np.std(our_rois)
    roi_stable = roi_std < 0.15
    logger.info(
        f"    ROI StdDev: {roi_std:.4f} ({'✅ Stable' if roi_stable else '⚠️ Volatile'})"
    )

    return {
        "rounds": rounds,
        "roi_stability": round(roi_std, 4),
        "roi_stable": roi_stable,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("🚀 PHASE 11: REAL-TIME MARKET & SIGNAL REALISM")
    logger.info("=" * 65)

    logger.info("Loading data...")
    X, y_ctr, y_cvr, prices = prepare_data()
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    biz = task1_business_metrics(X, y_ctr, y_cvr, prices)
    pareto = task2_pareto(X, y_ctr, y_cvr, prices)
    gate = task3_adaptive_gate(X, y_ctr, y_cvr, prices)
    delay = task4_delayed_feedback(X, y_ctr, y_cvr, prices)
    game = task5_game_theory(X, y_ctr, y_cvr, prices)

    logger.info("\n" + "=" * 65)
    logger.info("PHASE 11 COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  RPM Replay: ${biz['rpm_replay']:.2f}, DR RPM: ${biz['rpm_dr']:.2f}")
    logger.info(
        f"  Pareto points: {len(pareto['pareto'])}, Best profit: {pareto['recommended']['profit']:.0f}"
    )
    logger.info(
        f"  48h Gate: Util={gate['final_util']:.1%}, ROI={gate['final_roi']:.4f}"
    )
    logger.info(f"  Delay bias reduction: CTR {biz.get('uplift_pct', 'N/A')}%")
    logger.info(f"  Game equilibrium: ROI stable={game['roi_stable']}")

    return {
        "business": biz,
        "pareto": pareto,
        "gate": gate,
        "delay": delay,
        "game": game,
    }


if __name__ == "__main__":
    results = main()
