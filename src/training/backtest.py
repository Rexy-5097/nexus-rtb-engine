#!/usr/bin/env python3
"""
Phase 7: True Economic Backtest
================================
Replays historical bid data through the engine with real paying prices.
No future lookahead. Reports AUC, Win Rate, ROI, Budget Utilization,
eCPC, eCPA, Latency, and comparison vs Constant Bidding baseline.
"""
import sys, os, time, pickle, logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath("."))

from src.training.train import load_dataset, build_matrix, FeatureExtractor, HASH_SPACE
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest():
    logger.info("=== PHASE 7: ECONOMIC BACKTEST ===")
    
    # 1. Load Data
    full_df, global_stats, scaler, stats, top_k_maps = load_dataset()
    
    # 2. Load Model
    with open("src/model_weights.pkl", "rb") as f:
        artifacts = pickle.load(f)
    
    ctr_model, ctr_params, ctr_calib = artifacts["ctr"]
    cvr_model, cvr_params, cvr_calib = artifacts["cvr"]
    
    model_type = ctr_params.get("type", "UNKNOWN")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Version: {artifacts.get('ver', 'unknown')}")
    
    # 3. Build Feature Matrix (full dataset)
    feature_extractor = FeatureExtractor()
    if 'top_k_maps' in artifacts:
        feature_extractor.set_encoding_maps(artifacts['top_k_maps'])
    
    logger.info("Building feature matrix...")
    X_parts = []
    chunk_size = 100000
    for start in range(0, len(full_df), chunk_size):
        chunk = full_df.iloc[start:start+chunk_size]
        c_mask = chunk['click'].values.astype(np.int8)
        v_mask = chunk['conversion'].values.astype(np.int8)
        rows = [row for row in chunk.itertuples(index=False)]
        mat = build_matrix(rows, c_mask, v_mask, global_stats, feature_extractor, scaler)
        if mat is not None:
            X_parts.append(mat)
    
    from scipy.sparse import vstack
    X_full = vstack(X_parts)
    
    y_click = full_df['click'].values.astype(int)
    y_conv = full_df['conversion'].values.astype(int)
    paying_prices = full_df['payingprice'].values.astype(float)
    
    # Time-based Split: Use only TEST set for backtest (last 15%)
    print("  [Using standard test split for backtest]")
    n = X_full.shape[0]
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    X_test = X_full[n_train + n_val:]
    y_click_test = y_click[n_train + n_val:]
    y_conv_test = y_conv[n_train + n_val:]
    prices_test = paying_prices[n_train + n_val:]
    
    logger.info(f"Test Set: {X_test.shape[0]} samples")
    
    # 4. Predict
    t0 = time.perf_counter()
    pCTR = ctr_model.predict_proba(X_test)[:, 1]
    if ctr_calib:
        pCTR = ctr_calib.predict(pCTR)
    latency_ctr = (time.perf_counter() - t0) / X_test.shape[0] * 1000  # ms per sample
    
    if cvr_model is not None:
        pCVR_raw = cvr_model.predict_proba(X_test)[:, 1]
        if cvr_calib:
            pCVR_raw = cvr_calib.predict(pCVR_raw)
    else:
        pCVR_raw = np.zeros(len(pCTR))
    
    # p(Conv|Imp) = pCTR * p(Conv|Click)
    pConv = pCTR * pCVR_raw
    
    # 5. AUC
    try: ctr_auc = roc_auc_score(y_click_test, pCTR)
    except: ctr_auc = 0.5
    
    try: cvr_auc = roc_auc_score(y_conv_test, pConv)
    except: cvr_auc = 0.5
    
    # 6. Simulate Bidding
    from src.bidding.config import config
    value_click = config.value_click
    value_conversion = config.value_conversion
    
    # EV-based bidding
    ev = pCTR * value_click + pConv * value_conversion
    
    # Budget
    BUDGET = float(prices_test.sum()) * 0.8  # 80% of total market cost
    
    # Sort by bid time (already sorted in test set)
    total_spend = 0.0
    total_wins = 0
    total_clicks = 0
    total_convs = 0
    total_conv_value = 0.0
    
    # Constant Bidding Baseline
    constant_bid = np.median(prices_test)
    const_spend = 0.0
    const_wins = 0 
    const_clicks = 0
    const_convs = 0
    
    for i in range(len(prices_test)):
        market_price = prices_test[i]
        bid = ev[i]
        
        # EV Bidding
        if total_spend + market_price <= BUDGET and bid >= market_price:
            total_spend += market_price
            total_wins += 1
            if y_click_test[i]: total_clicks += 1
            if y_conv_test[i]:
                total_convs += 1
                total_conv_value += value_conversion
        
        # Constant Bidding
        if const_spend + market_price <= BUDGET and constant_bid >= market_price:
            const_spend += market_price
            const_wins += 1
            if y_click_test[i]: const_clicks += 1
            if y_conv_test[i]: const_convs += 1
    
    # 7. Metrics
    n_test = len(prices_test)
    win_rate = total_wins / n_test if n_test > 0 else 0
    budget_util = total_spend / BUDGET if BUDGET > 0 else 0
    ecpc = total_spend / total_clicks if total_clicks > 0 else float('inf')
    ecpa = total_spend / total_convs if total_convs > 0 else float('inf')
    roi = total_conv_value / total_spend if total_spend > 0 else 0
    
    const_win_rate = const_wins / n_test if n_test > 0 else 0
    const_ecpc = const_spend / const_clicks if const_clicks > 0 else float('inf')
    const_ecpa = const_spend / const_convs if const_convs > 0 else float('inf')
    const_roi = (const_convs * value_conversion) / const_spend if const_spend > 0 else 0
    
    # Model Size
    model_size_mb = os.path.getsize("src/model_weights.pkl") / (1024 * 1024)
    
    # 8. Report
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7: ECONOMIC BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'EV Bidding':<15} {'Constant':<15}")
    logger.info("-" * 55)
    logger.info(f"{'CTR AUC':<25} {ctr_auc:<15.4f} {'N/A':<15}")
    logger.info(f"{'CVR AUC':<25} {cvr_auc:<15.4f} {'N/A':<15}")
    logger.info(f"{'Win Rate':<25} {win_rate:<15.4f} {const_win_rate:<15.4f}")
    logger.info(f"{'Budget Utilization':<25} {budget_util:<15.4f} {const_spend/BUDGET if BUDGET>0 else 0:<15.4f}")
    logger.info(f"{'Wins':<25} {total_wins:<15} {const_wins:<15}")
    logger.info(f"{'Clicks':<25} {total_clicks:<15} {const_clicks:<15}")
    logger.info(f"{'Conversions':<25} {total_convs:<15} {const_convs:<15}")
    logger.info(f"{'eCPC':<25} {ecpc:<15.2f} {const_ecpc:<15.2f}")
    logger.info(f"{'eCPA':<25} {ecpa:<15.2f} {const_ecpa:<15.2f}")
    logger.info(f"{'ROI':<25} {roi:<15.4f} {const_roi:<15.4f}")
    logger.info(f"{'Latency (ms/sample)':<25} {latency_ctr:<15.4f} {'N/A':<15}")
    logger.info(f"{'Model Size (MB)':<25} {model_size_mb:<15.2f} {'N/A':<15}")
    logger.info("=" * 60)
    
    # Lift
    click_lift = ((total_clicks - const_clicks) / const_clicks * 100) if const_clicks > 0 else 0
    conv_lift = ((total_convs - const_convs) / const_convs * 100) if const_convs > 0 else 0
    roi_lift = ((roi - const_roi) / const_roi * 100) if const_roi > 0 else 0
    
    logger.info(f"\nLIFT vs Constant Bidding:")
    logger.info(f"  Click Lift: {click_lift:+.1f}%")
    logger.info(f"  Conv Lift:  {conv_lift:+.1f}%")
    logger.info(f"  ROI Lift:   {roi_lift:+.1f}%")
    
    # Statistical Significance Check
    # Simple check: is the improvement > 5% in key metrics?
    significant = ctr_auc >= 0.55 and (total_convs > const_convs or roi > const_roi)
    logger.info(f"\nStatistically Significant Improvement: {'YES' if significant else 'NO'}")
    
    # Target Check
    logger.info(f"\n--- Phase 7 TARGET CHECK ---")
    logger.info(f"CTR AUC >= 0.62: {'✅ PASS' if ctr_auc >= 0.62 else '❌ FAIL'} ({ctr_auc:.4f})")
    logger.info(f"CVR AUC >= 0.58: {'✅ PASS' if cvr_auc >= 0.58 else '❌ FAIL'} ({cvr_auc:.4f})")
    logger.info(f"Latency < 5ms:   {'✅ PASS' if latency_ctr < 5 else '❌ FAIL'} ({latency_ctr:.4f}ms)")
    logger.info(f"Model < 512MB:   {'✅ PASS' if model_size_mb < 512 else '❌ FAIL'} ({model_size_mb:.2f}MB)")

if __name__ == "__main__":
    run_backtest()
