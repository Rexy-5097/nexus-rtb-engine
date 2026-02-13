import numpy as np
from typing import List, Dict

def calculate_psi(expected: List[float], actual: List[float], bucket_type: str = "bins", buckets: int = 10, axis: int = 0) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    
    Args:
        expected: Reference distribution (e.g., training data scores).
        actual: Current distribution (e.g., production traffic scores).
        buckets: Number of quantiles/bins.
    
    Returns:
        PSI value. 
        - < 0.1: No significant shift.
        - 0.1 - 0.2: Moderate shift.
        - > 0.2: Significant shift.
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if bucket_type == 'bins':
        breakpoints = np.linspace(0, 1, buckets + 1)
    else:
        # Quantiles might be better if data is not uniform
        try:
             breakpoints = np.percentile(expected, breakpoints)
        except:
             breakpoints = np.linspace(0, 1, buckets + 1)

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return value

    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])

    return psi_value

class DriftDetector:
    """Stateful drift detector."""
    
    def __init__(self, reference_data: List[float] = None, window_size: int = 1000):
        self.reference = reference_data if reference_data else []
        self.window = []
        self.window_size = window_size
        
    def add(self, p_ctrs: float) -> float:
        """Add value and return current PSI."""
        self.window.append(p_ctrs)
        if len(self.window) > self.window_size:
            self.window.pop(0)
            
        if len(self.reference) < 100 or len(self.window) < 100:
            return 0.0
            
        return calculate_psi(self.reference, self.window)
