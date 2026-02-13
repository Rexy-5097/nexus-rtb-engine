from typing import Optional
from src.bidding.config import config

class Validator:
    """
    Input validation utilities to prevent DoS attacks and malformed requests.
    """
    
    @staticmethod
    def sanitize_string(s: Optional[str], max_len: int = config.max_string_length, default: str = "unknown") -> str:
        """
        Sanitize and truncate string inputs.
        Prevents memory exhaustion attacks via excessively long strings.
        """
        if not s:
            return default
        
        # Strip whitespace and truncate
        s = str(s).strip()
        if len(s) > max_len:
            # Log warning in production here
            return s[:max_len]
        
        if not s or s.lower() == "nan":
            return default
            
        return s

    @staticmethod
    def parse_floor_price(floor_str: Optional[str]) -> int:
        """
        Parse floor price safely. Returns 0 if invalid or missing.
        """
        if not floor_str:
            return 0
        try:
            # Handle float strings like "99.99" by converting to int (floor)
            return int(float(floor_str))
        except (ValueError, TypeError):
            return 0
