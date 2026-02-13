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

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the CTR/CVR prediction models."""
    feature_bits: int = 18
    hash_space: int = 2 ** 18
    # Default intercepts (fail-safe values ~1.8% probability)
    default_intercept_ctr: float = -4.0
    default_intercept_cvr: float = -4.0
    
    # Advertiser N-values (Conversion importance)
    n_map: Dict[str, int] = field(default_factory=lambda: {
        "1458": 0,  # Local e-commerce
        "3358": 2,  # Software
        "3386": 0,  # Global e-commerce
        "3427": 0,  # Oil
        "3476": 10, # Tire
    })

@dataclass(frozen=True)
class EngineConfig:
    """Master configuration for the Bidding Engine."""
    max_bid_price: int = 300
    min_bid_price: int = 0
    # Reject bids if EV < quality_threshold * avg_ev
    quality_threshold: float = 0.40
    # Cap bid multiplier relative to market price
    max_market_ratio: float = 3.0
    # Input validation
    max_string_length: int = 512

    pacing: PacingConfig = field(default_factory=PacingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

# Global singleton config instance
config = EngineConfig()
