import threading
import time
import logging
from typing import Dict, Tuple

from src.bidding.config import config

logger = logging.getLogger(__name__)

class PacingController:
    """
    Controls bid pacing using a PID-like feedback loop to ensure budget longevity.
    Thread-safe implementation for concurrent bidding environments.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._requests_seen = 0
        self._estimated_spend = 0.0
        
        # Local cache of config for speed
        self.conf = config.pacing
        self.total_budget = self.conf.total_budget
        self.expected_requests = self.conf.expected_requests
        self.estimated_win_rate = self.conf.estimated_win_rate

    def get_pacing_factor(self) -> float:
        """
        Get the current pacing factor. 
        Note: Simple version returns standard factor, 
        update() returns the dynamic one.
        """
        # Ideally this would return the current dynamic factor.
        # For this implementation, we calculate it on the fly in update() 
        # or we could store a `self.current_factor`.
        # Let's delegate to update() logic validation or return default.
        return 1.0

    def record_bid(self, price: float):
        """Record a bid."""
        self.update(price)

    def update(self, raw_bid: float) -> Tuple[float, float]:
        """
        Update pacing state with a new potential bid and return the pacing factor.
        
        Returns:
            Tuple[pacing_factor, estimated_spend_increment]
        """
        with self._lock:
            self._requests_seen += 1
            
            # --- Pacing Logic ---
            ideal_spend = (self._requests_seen / self.expected_requests) * self.total_budget
            actual_spend = self._estimated_spend
            
            # Determine pacing factor based on spend velocity
            if actual_spend > ideal_spend * self.conf.overspend_threshold:
                factor = self.conf.factor_cool_down
            elif actual_spend < ideal_spend * self.conf.underspend_threshold:
                factor = self.conf.factor_speed_up
            else:
                factor = self.conf.factor_steady
            
            # Record the bid's impact on estimated budget
            spend_impact = raw_bid * factor * self.estimated_win_rate
            self._estimated_spend += spend_impact
            
            return factor, self._estimated_spend
            
    def get_stats(self) -> Dict[str, float]:
        """Return current pacing stats for observability."""
        with self._lock:
            return {
                "requests_seen": float(self._requests_seen),
                "estimated_spend": self._estimated_spend,
                "burn_rate": self._estimated_spend / (self._requests_seen + 1) if self._requests_seen > 0 else 0.0
            }
