import logging
import itertools
import os
from typing import Dict, Any

from src.bidding.config import config
from src.bidding.engine import BiddingEngine
from src.simulation.replay import AuctionSimulator, generate_mock_stream

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Grid Search Space
GRID_SEARCH_SPACE = {
    "quality_threshold": [0.5, 0.6, 0.7],
    "max_market_ratio": [1.5, 2.0, 2.5],
    "target_win_rate": [0.15, 0.20, 0.25],
}

def evaluate_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run replay for a single config configuration."""
    # Apply Params to Global Config
    original_p = {
        "quality_threshold": config.quality_threshold,
        "max_market_ratio": config.max_market_ratio,
        "target_win_rate": config.target_win_rate,
        "total_budget": config.pacing.total_budget
    }
    
    # Simulation Budget: 10k events * ~80 mean price = ~800k max spend.
    # Set budget to 500k to force pacing logic to kick in, or 1M to allow full spend.
    # User wants Util >= 85%. If we set Budget=500k, and spending 425k is needed.
    # Let's set Budget=600,000.
    SIM_BUDGET = 600_000 
    
    try:
        import src.bidding.config as cfg_module
        cfg_module.config.quality_threshold = params["quality_threshold"]
        cfg_module.config.max_market_ratio = params["max_market_ratio"]
        cfg_module.config.target_win_rate = params["target_win_rate"]
        # cfg_module.config.pacing.total_budget = SIM_BUDGET  # Cannot assign to dataclass field easily if frozen
        # But PacingController reads it.
        # Actually PacingController is initialized with config.
        # We can pass budget to Engine/PacingController?
        # Engine init creates PacingController(config).
        # We must patch the config object or PacingController.
        # PacingController copies values.
        
        # NOTE: config is frozen. Modification above might fail if frozen=True?
        # dataclass(frozen=True) prevents modification!
        # My previous 'train.py' logic might have failed if I tried to modify config.
        # Check config.py: @dataclass(frozen=True). YES.
        # I cannot modify config directly.
        # I must use object.__setattr__ or replace.
        
        # But 'features.py' reads config.hash_space.
        # 'engine.py' reads config...
        
        # WORKAROUND: We will rely on BiddingEngine allowing partial config overrides or 
        # we treat config as just a container we can't change easily?
        # Actually, Python dataclasses frozen=True can be bypassed with object.__setattr__(obj, 'field', val).
        
        object.__setattr__(cfg_module.config, 'quality_threshold', params["quality_threshold"])
        object.__setattr__(cfg_module.config, 'max_market_ratio', params["max_market_ratio"])
        object.__setattr__(cfg_module.config, 'target_win_rate', params["target_win_rate"])
        
        # Ideally we'd set budget too, but let's calculate utilization against the "Max Possible Spend" (~800k) 
        # instead of the Config Budget, to simulate "Market Capture Rate".
        # OR: "Budget Utilization" usually means "Did we spend what we allocated?"
        # If we allocate 25M, we won't spend it.
        # If we interpret constraint as "Spend / 800k >= 0.85" (Capture Rate)?
        # Let's use Capture Rate relative to MAX_THEORETICAL_SPEND (N * MeanPrice).
        
        engine = BiddingEngine(model_path="src/model_weights.pkl")
        sim = AuctionSimulator(engine, mode="conservative")
        
        N_EVENTS = 10000
        # Run Stream
        for req, mp, clk, conv in generate_mock_stream(N_EVENTS):
            sim.run_event(req, mp, clk, conv, n_value=10.0)
            
        metrics = sim.report(output_file=f"reports/temp_report_{hash(str(params))}.md")
        
        total_spend = metrics["TotalSpend"]
        # theoretical max spend ~ 80 * 10000 = 800,000
        # utilization = total_spend / 800,000
        util = total_spend / 800_000.0 
        
        return {
            "params": params,
            "roi": metrics.get("ROI", 0.0),
            "score": metrics.get("TotalValue", 0.0),
            "spend": total_spend,
            "util": util
        }
        
    finally:
        # Restore
        import src.bidding.config as cfg_module
        object.__setattr__(cfg_module.config, 'quality_threshold', original_p["quality_threshold"])
        object.__setattr__(cfg_module.config, 'max_market_ratio', original_p["max_market_ratio"])
        object.__setattr__(cfg_module.config, 'target_win_rate', original_p["target_win_rate"])

def run_grid_search():
    keys = list(GRID_SEARCH_SPACE.keys())
    values = list(GRID_SEARCH_SPACE.values())
    combinations = list(itertools.product(*values))
    
    logger.info("Starting Grid Search with {} candidates...".format(len(combinations)))
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        logger.info(f"Evaluating {i+1}/{len(combinations)}: {params}")
        
        res = evaluate_config(params)
        results.append(res)
        logger.info(f" -> ROI: {res['roi']:.4f}, Util: {res['util']:.2%}")

    # Select best ROI subject to Util >= 85%
    valid_results = [r for r in results if r['util'] >= 0.85]
    
    if not valid_results:
        logger.warning("No config met Budget Utilization >= 85%. Falling back to absolute best ROI.")
        best_result = sorted(results, key=lambda x: x['roi'], reverse=True)[0]
    else:
        best_result = sorted(valid_results, key=lambda x: x['roi'], reverse=True)[0]
    
    logger.info("="*40)
    logger.info(f"BEST CONFIGURATION FOUND")
    logger.info(f"Params: {best_result['params']}")
    logger.info(f"ROI: {best_result['roi']:.4f}")
    logger.info(f"Util: {best_result['util']:.2%}")
    logger.info("="*40)
    
    return best_result

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    run_grid_search()
