import threading
import time
import logging
from typing import Dict, Tuple

from src.bidding.config import config

logger = logging.getLogger(__name__)

class PacingController:
    """
    Controls bid pacing using a PID feedback loop to ensure budget longevity and safety.
    
    Features:
    - Thread-safe state management.
    - Hard budget caps (Daily, Hourly, Minute).
    - PID control for smooth spend velocity.
    - Market shock / Traffic surge dampening.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # State
        self._total_spent = 0.0
        self._requests_seen = 0
        self._last_update_time = time.time()
        
        # PID State
        self._error_integral = 0.0
        self._last_error = 0.0
        self._current_factor = config.alpha_initial  # Start with configured Alpha
        
        # Rate Limiting (Token Bucket -ish)
        self._hourly_spend = 0.0
        self._minute_spend = 0.0
        self._last_hour_reset = time.time()
        self._last_minute_reset = time.time()

        # Cache config
        self.conf = config.pacing
        self.engine_conf = config

    def can_bid(self) -> bool:
        """
        Check if we are allowed to bid based on hard budget caps.
        This is a 'Circuit Breaker' check.
        """
        with self._lock:
            # 1. Daily Hard Cap
            if self._total_spent >= self.conf.max_daily_spend:
                return False

            # 2. Hourly Soft Cap
            if self._hourly_spend >= self.conf.max_hourly_spend:
                if time.time() - self._last_hour_reset < 3600:
                    return False
                else:
                    self._reset_hourly()

            # 3. Surge Protection (Minute Cap)
            if self._minute_spend >= self.conf.max_minute_spend:
                if time.time() - self._last_minute_reset < 60:
                    return False
                else:
                    self._reset_minute()
                    
            return True

    def update(self, bid_price: float) -> float:
        """
        Update pacing state after a bid decision (even if lost, we track potential spend).
        Returns the **dynamic alpha** (pacing factor) for the NEXT bid.
        
        Args:
            bid_price: The generated bid price (before alpha application? No, usually tracked after).
                       Here we track *committed* spend or *expected* spend.
                       For RTB, we usually track 'win' spend, but we don't know if we won yet.
                       So we track 'potential' spend or rely on win-rate feedback.
                       
                       Simplification: We use a PID to target a 'Spend Rate'.
        """
        now = time.time()
        
        with self._lock:
            # Reset counters if needed
            if now - self._last_hour_reset > 3600: self._reset_hourly()
            if now - self._last_minute_reset > 60: self._reset_minute()

            self._requests_seen += 1
            
            # --- PID Control ---
            # Target: Uniform spend rate over time
            # Target Rate = Budget / Remaining Time
            # We approximate: Target Spending per Request = Total Budget / Expected Requests
            
            target_spend_per_req = self.conf.total_budget / self.conf.expected_requests
            
            # Actual Spend Rate (Smoothed)
            # We approximate actual spend by (Total Spend / Requests)
            # Note: This implies we get win notifications. 
            # If we don't get win notifications, we assume a Win Rate.
            estimated_win_spend = bid_price * self.conf.estimated_win_rate
            
            # Update accumulators
            self._total_spent += estimated_win_spend
            self._hourly_spend += estimated_win_spend
            self._minute_spend += estimated_win_spend
            
            current_spend_rate = self._total_spent / max(1, self._requests_seen)
            
            # Error = Target - Actual
            # Positive error (Target > Actual) -> Underspending -> Increase Alpha
            # Negative error (Target < Actual) -> Overspending -> Decrease Alpha
            error = target_spend_per_req - current_spend_rate
            
            # PID Terms
            dt = now - self._last_update_time
            if dt <= 0: dt = 0.001
            
            # P: Proportional
            p_term = self.conf.pacing_k_p * error
            
            # I: Integral (Accumulated error)
            # Clamp integral to prevent windup
            self._error_integral += error * dt
            self._error_integral = max(-10.0, min(10.0, self._error_integral))
            i_term = self.conf.pacing_k_i * self._error_integral
            
            # D: Derivative (Rate of change of error)
            d_error = (error - self._last_error) / dt
            d_term = self.conf.pacing_k_d * d_error
            
            # Output Adjustment
            delta = p_term + i_term + d_term
            
            # Apply to current factor
            self._current_factor += delta
            
            # --- Safety Clamps ---
            # Clamp Alpha to safe bounds [0.1, 2.0]
            self._current_factor = max(
                self.engine_conf.alpha_min, 
                min(self.engine_conf.alpha_max, self._current_factor)
            )
            
            # State Rotation
            self._last_error = error
            self._last_update_time = now
            
            return self._current_factor

    def _reset_hourly(self):
        self._hourly_spend = 0.0
        self._last_hour_reset = time.time()

    def _reset_minute(self):
        self._minute_spend = 0.0
        self._last_minute_reset = time.time()

    def get_stats(self) -> Dict[str, float]:
        """Return observability metrics."""
        with self._lock:
            return {
                "alpha": self._current_factor,
                "total_spent": self._total_spent,
                "requests": float(self._requests_seen),
                "hourly_spend": self._hourly_spend,
                "minute_spend": self._minute_spend
            }
