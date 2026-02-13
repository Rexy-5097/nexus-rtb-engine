import threading
from typing import Tuple
from src.bidding.config import config

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
            # We assume we win 20% of bids we place (Estimated Win Rate)
            # and that we pay roughly the bid amount (conservative estimate for pacing)
            # Note: In 2nd price auction we pay less, but using raw_bid acts as a safety buffer.
            spend_impact = raw_bid * factor * self.estimated_win_rate
            self._estimated_spend += spend_impact
            
            return factor, self._estimated_spend
            
    def get_stats(self) -> dict:
        """Return current pacing stats for observability."""
        with self._lock:
            return {
                "requests_seen": self._requests_seen,
                "estimated_spend": self._estimated_spend,
                "burn_rate": self._estimated_spend / (self._requests_seen + 1)
            }
