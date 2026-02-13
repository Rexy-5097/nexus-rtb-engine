from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class PacingConfig:
    """Configuration for the PID-based adaptive pacing controller."""
    expected_requests: int = 25_000_000
    total_budget: int = 25_000_000
    estimated_win_rate: float = 0.20
    
    # PID thresholds
    overspend_threshold: float = 1.0     # When actual > ideal * 1.0
    underspend_threshold: float = 0.9    # When actual < ideal * 0.9
    
    # Pacing factors
    factor_cool_down: float = 0.80
    factor_speed_up: float = 1.15
    factor_steady: float = 1.00
    
    # PID factors
    pacing_k_p: float = 0.05
    pacing_k_i: float = 0.01
    pacing_k_d: float = 0.005
    
    # Circuit Breakers
    max_daily_spend: int = 25_000_000  # Hard Cap
    max_hourly_spend: int = 2_000_000  # Soft Cap
    max_minute_spend: int = 50_000     # Surge Protection

    # Feature Flags / Canary
    strategy_version: str = "v2_ev_safe" # v1=Stable, v2=Experimental
    experiment_traffic: float = 0.1 # 10% traffic to v2

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the CTR/CVR prediction models."""
    feature_bits: int = 18
    hash_space: int = 2 ** 18
    # Default intercepts (fail-safe values ~0.1% probability)
    # Lowered from -4.0 to -7.0 for "Fail-Closed" / Conservative start
    default_intercept_ctr: float = -7.0
    default_intercept_cvr: float = -7.0
    
    # Advertiser N-values (Conversion importance)
    n_map: Dict[str, int] = field(default_factory=lambda: {
        "1458": 0,  # Local e-commerce
        "3358": 2,  # Software
        "3386": 0,  # Global e-commerce
        "3427": 0,  # Oil
        "3476": 10, # Tire
    })

@dataclass(frozen=False)
class EngineConfig:
    """Master configuration for the Bidding Engine."""
    # Bidding Limits
    max_bid_price: int = 300
    min_bid_price: int = 1
    
    # Value Estimation
    value_click: float = 50.0  # Base value per click
    value_conversion: float = 500.0 # Base value per conversion
    
    # Dynamic Alpha (Bid Shading)
    alpha_initial: float = 0.8
    alpha_min: float = 0.1
    alpha_max: float = 2.0
    
    # Target Win Rate (18%)
    target_win_rate: float = 0.18
    # Valuation (for ROI calculation)
    # Increased for Phase 5 Log-Normal Market (Mean Price ~80)
    value_click: float = 500.0
    value_conversion: float = 5000.0
    
    # ROI Guards
    max_cpa: float = 150.0 # Do not bid if predicted CPA > this
    
    # Bidding Control
    # quality_threshold: Optimized via Grid Search (Phase 5)
    quality_threshold: float = 0.50
    # Cap bid multiplier relative to market price
    max_market_ratio: float = 2.0
    
    # Target Win Rate (Optimized)
    target_win_rate: float = 0.202
    # Input validation
    max_string_length: int = 512

    pacing: PacingConfig = field(default_factory=PacingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Risk Control & Drift
    psi_threshold: float = 0.2
    risk_mode: bool = True
    cvar_alpha: float = 0.05 # Bottom 5% tail risk

# Global singleton config instance
config = EngineConfig()
