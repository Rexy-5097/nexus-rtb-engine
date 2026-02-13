
import logging
import sys
import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import pickle

# Setup path
sys.path.append(os.getcwd())

from src.training.train import load_dataset, FeatureExtractor, build_matrix
from src.bidding.config import config
from src.bidding.engine import BiddingEngine
from src.bidding.schema import BidRequest
from src.bidding.features import FeatureExtractor as RealFeatureExtractor
from src.bidding.model import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIM_BUDGET = 400000

# Global caches for pre-computed scores
SCORE_CACHE = {} # bidId -> {'ctr': float, 'cvr': float, 'price': float}

class MockFeatureExtractor:
    def extract(self, req, scaler, stats):
        # Pass bidId as "features"
        return req.bidId

class MockModelLoader(ModelLoader):
    def __init__(self, path):
        # Skip loading
        self.model_loaded = True
        self.scaler = {}
        self.stats = {}
        self.n_map = {}
        self.adv_priors = {}
        self.lgb_price = "mock" # Just not None

    def predict_ctr(self, bid_id):
        return SCORE_CACHE.get(bid_id, {}).get('ctr', 0.001)
        
    def predict_cvr(self, bid_id):
        return SCORE_CACHE.get(bid_id, {}).get('cvr', 0.0)
        
    def predict_market_price(self, bid_id):
        return SCORE_CACHE.get(bid_id, {}).get('price', 50.0)
        
    def get_stats(self, adv_id):
        # We need stats for Quality Gate.
        # REAL_STATS contains rolling lists, not the aggregation model.py expects
        # So we return safe defaults.
        return {"avg_mp": 50.0, "avg_ev": 0.001}

REAL_STATS = {}

def precompute_scores(full_df):
    """
    Load model, build matrix, predict all, populate SCORE_CACHE.
    """
    logger.info("Pre-computing scores...")
    
    # Load Real Model
    with open("src/model_weights.pkl", "rb") as f:
        data = pickle.load(f)
        
    # Stats
    global REAL_STATS
    REAL_STATS = data.get("stats", {})
    scaler = data.get("scaler", {})
    
    # Models
    # We assume LGBM for speed
    ctr_obj, _, _ = data["ctr"]
    cvr_obj, _, _ = data["cvr"]
    price_obj = data.get("price_model")
    
    # Check if linear or lgb
    # Assuming standard flow used LGB?
    # If linear, we need to handle that.
    # But for optimization, we just need ANY valid scores.
    # If Linear, we can compute dot product?
    # Or just use the predict function from model_loader?
    
    # Proper way: Use ModelLoader to predict in batch?
    # ModelLoader abstracts single row.
    # We want batch predict.
    
    # Build Matrix X_full
    logger.info("Building Feature Matrix...")
    fe = FeatureExtractor() # Training FE
    
    # We process in chunks to avoid massive memory usage if needed
    # But for 10k rows, just list comprehension
    rows = [row for row in full_df.itertuples(index=False)]
    
    # Dummy masks
    dummy_y = np.zeros(len(rows))
    
    # build_matrix expects global_stats, fe, scaler
    # global_stats is usually from Pass 1.
    # We can fetch it from load_dataset return or re-compute?
    # load_dataset returns global_stats.
    # We need it.
    
    # X_full is CSR matrix
    # If using train.build_matrix, we need global_stats.
    pass

def setup_environment():
    full_df, global_stats, scaler, stats = load_dataset()
    
    # Fix missing columns if needed
    if 'userTags' not in full_df.columns:
        full_df['userTags'] = "0"
        
    with open("src/model_weights.pkl", "rb") as f:
        artifacts = pickle.load(f)
        
    global REAL_STATS
    REAL_STATS = stats
    
    # Prepare Models
    ctr_mdl, _, _ = artifacts["ctr"]
    cvr_mdl, _, _ = artifacts["cvr"]
    price_mdl = artifacts.get("price_model")
    
    fe = RealFeatureExtractor()
    
    logger.info("Building full matrix...")
    # build_matrix takes (rows, y_c, y_v, global_stats, fe, scaler)
    rows = [row for row in full_df.itertuples(index=False)]
    y_dummy = np.zeros(len(rows), dtype=np.int8)
    
    X_full = build_matrix(rows, y_dummy, y_dummy, global_stats, fe, scaler)
    
    logger.info("Batch Predicting...")
    # CTR
    if hasattr(ctr_mdl, "predict_proba"):
        p_ctr = ctr_mdl.predict_proba(X_full)[:, 1]
    else:
        # Linear (coef_, intercept_)
        # p = 1 / (1 + exp(- (X @ w + b)))
        w = ctr_mdl.coef_.T
        b = ctr_mdl.intercept_
        logits = X_full.dot(w) + b
        p_ctr = 1.0 / (1.0 + np.exp(-logits))
        p_ctr = p_ctr.flatten()

    # CVR
    if cvr_mdl:
        if hasattr(cvr_mdl, "predict_proba"): # LR
             p_cvr = cvr_mdl.predict_proba(X_full)[:, 1]
        elif hasattr(cvr_mdl, "predict"): # LGB Regressor/Classifier
             # LGBClassifier predict_proba, LGBRegressor predict
             try:
                 p_cvr = cvr_mdl.predict_proba(X_full)[:, 1]
             except:
                 p_cvr = cvr_mdl.predict(X_full)
    else:
        p_cvr = np.zeros(len(rows))
        
    # Price
    if price_mdl:
        log_price = price_mdl.predict(X_full)
        p_price = np.expm1(log_price)
    else:
        p_price = np.full(len(rows), 50.0)
        
    logger.info("Populating Cache...")
    for i, bid_id in enumerate(full_df['bidid']):
        # Simulation loop constructs bidId as "sim_{i}" or keeps original?
        # Simulation loop constructed "sim_{i}".
        # But we need to map row index to score.
        # We can use "sim_{i}" as key.
        key = f"sim_{i}"
        SCORE_CACHE[key] = {
            'ctr': float(p_ctr[i]),
            'cvr': float(p_cvr[i]),
            'price': float(p_price[i])
        }
        
    return full_df

def run_simulation(params: Dict[str, float], full_df: pd.DataFrame) -> Dict[str, float]:
    # Override Config
    # Boost valuations to be competitive in Mock Market (Price ~80)
    # Breakeven: Val > Price/CTR ~ 80/0.05 = 1600.
    if "value_click" not in params:
        setattr(config, "value_click", 2000.0)
        setattr(config, "value_conversion", 20000.0)
        
    for k, v in params.items():
        setattr(config, k, v)
        
    # Reset Pacing (Mock Engine)
    # We create engine but Inject Mocks
    engine = BiddingEngine("src/model_weights.pkl") # Helper
    # Override components
    engine.feature_extractor = MockFeatureExtractor()
    engine.model_loader = MockModelLoader("src/model_weights.pkl")
    # Reset Pacing
    engine.pacing._total_budget = SIM_BUDGET
    engine.pacing._spent_budget = 0.0
    engine.pacing._win_history.clear()
    
    total_spend = 0.0
    total_wins = 0
    total_convs = 0.0
    
    # Iterate
    # We use row index to link to cache
    for i, row in full_df.iterrows():
        bid_id = f"sim_{i}"
        req = BidRequest(
            bidId=bid_id,
            timestamp=str(row['timestamp']),
            visitorId="v", userAgent="ua", ipAddress="1.2.3.4",
            region="US", city="NY", adExchange="nx",
            domain=row['domain'],
            url="u", anonymousURLID="a",
            adSlotID="s", adSlotWidth="300", adSlotHeight="250",
            adSlotVisibility=str(row['vis']),
            adSlotFormat=str(row['fmt']),
            adSlotFloorPrice="0",
            creativeID="c",
            advertiserId=row['advertiser'],
            userTags=getattr(row, 'userTags', '0')
        )
        
        # Engine Process
        resp = engine.process(req)
        
        if resp.bidPrice > 0:
            market_price = row['payingprice']
            target_wr = params.get("target_win_rate", config.target_win_rate)
            reserved = resp.bidPrice * target_wr
            
            if resp.bidPrice >= market_price:
                # Win
                cost = market_price
                total_spend += cost
                total_wins += 1
                total_convs += row['conversion']
                diff = cost - reserved
                with engine.pacing._lock:
                    engine.pacing._spent_budget += diff
                    if engine.pacing._win_history: engine.pacing._win_history[-1] = 1
            else:
                # Lose
                diff = 0.0 - reserved
                with engine.pacing._lock:
                    engine.pacing._spent_budget += diff
                    # History remains 0
                    
    util = total_spend / SIM_BUDGET
    roi = (total_convs * config.value_conversion) / (total_spend + 1e-6)
    
    return {"spend": total_spend, "convs": total_convs, "util": util, "roi": roi}

def optimize():
    full_df = setup_environment()
    
    search_space = []
    for _ in range(10):
        search_space.append({
            "target_win_rate": random.uniform(0.12, 0.22),
            "quality_threshold": random.uniform(0.5, 0.8),
            "max_market_ratio": random.uniform(1.2, 2.0)
        })
        
    results = []
    logger.info("Starting Simulation...")
    for p in search_space:
        try:
            res = run_simulation(p, full_df)
            res.update(p)
            results.append(res)
            logger.info(f"Result: Util={res['util']:.2%} Convs={res['convs']:.2f}")
        except Exception as e:
            logger.error(f"Sim Error: {e}")
            
    if results:
        best = max(results, key=lambda x: x['convs'] if x['util'] >= 0.85 else -1)
        logger.info(f"\nBest Params: {best}")
    else:
        logger.error("No results.")

if __name__ == "__main__":
    optimize()
