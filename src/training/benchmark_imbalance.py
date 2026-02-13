#!/usr/bin/env python3
import sys
import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Add src to path
sys.path.append(os.getcwd())

# Need to import helper function "build_matrix", "load_dataset" from train.py
# However, "load_dataset" is not exposed in the truncated version unless I restored it correctly.
# Assuming I restored it correctly in last step.
try:
    from src.training.train import load_dataset, FeatureExtractor, build_matrix, global_stats, scaler
except ImportError:
    # If explicit import fails, maybe relative? 
    # Or redefine minimal helpers.
    # Let's hope it works.
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark():
    logger.info("Loading Data...")
    
    # We call load_dataset which returns full_df etc.
    full_df, global_stats, scaler, stats = load_dataset()
    
    logger.info("Building Training Matrices...")
    feature_extractor = FeatureExtractor()
    
    X_parts = []
    y_ctr_parts = []
    
    chunk_size = 100000
    # Process full_df in chunks
    for start in range(0, len(full_df), chunk_size):
        chunk = full_df.iloc[start:start+chunk_size]
        c_mask = chunk['click'].values.astype(np.int8)
        v_mask = chunk['conversion'].values.astype(np.int8)
        
        rows = [row for row in chunk.itertuples(index=False)]
        
        # build_matrix expects global_stats to be passed
        # and feature_extractor and scaler
        mat = build_matrix(rows, c_mask, v_mask, global_stats, feature_extractor, scaler)
        
        if mat is not None:
            X_parts.append(mat)
            y_ctr_parts.append(c_mask)
            
    if not X_parts:
        sys.exit("No data")
            
    X_full = vstack(X_parts)
    # y_full = np.concatenate(y_ctr_parts)
    # Use list comprehension to flatten if needed, or concatenate
    y_full = np.concatenate(y_ctr_parts)
    
    # Split (Train/Val/Test)
    n = X_full.shape[0]
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    
    X_tr = X_full[:n_tr]
    y_tr = y_full[:n_tr]
    
    X_va = X_full[n_tr:n_tr+n_va]
    y_va = y_full[n_tr:n_tr+n_va]
    
    pos_rate = y_tr.mean()
    scale_pos = (1.0 - pos_rate) / (pos_rate + 1e-6)
    logger.info(f"Data Split: Train={n_tr} Val={n_va} PosRate={pos_rate:.4f} Scale={scale_pos:.2f}")
    
    # Strategies
    strategies = [
        {"name": "LightGBM-Baseline", "params": {}},
        {"name": "LightGBM-ScalePosWeight", "params": {"scale_pos_weight": scale_pos}},
        {"name": "LightGBM-IsUnbalance", "params": {"is_unbalance": True}},
        {"name": "LightGBM-GOSS", "params": {"boosting_type": "goss"}}, 
    ]
    
    results = []
    
    for stra in strategies:
        try:
            name = stra["name"]
            spec_params = stra["params"]
            logger.info(f"--- Benchmarking {name} ---")
            
            p = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'n_estimators': 100,
                'verbose': -1,
            }
            p.update(spec_params)
            
            model = lgb.LGBMClassifier(**p)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(10)]
            )
            
            # Predict
            probs = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, probs)
            logger.info(f"{name} Val AUC: {auc:.4f}")
            results.append({"strategy": name, "auc": auc})
            
        except Exception as e:
            logger.error(f"{name} Failed: {e}")
            results.append({"strategy": name, "auc": 0.0, "error": str(e)})
            
    # Report
    logger.info("\n=== Imbalance Benchmark Results ===")
    res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    logger.info("\n" + str(res_df))
    
    # Save to file
    res_df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    benchmark()
