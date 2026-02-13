import threading
from typing import Dict

class BudgetCoordinator:
    """
    Simulates a centralized budget coordinator (like Redis).
    Manages atomic budget allocation across multiple distributed bidder instances.
    """
    
    def __init__(self, total_budget: float):
        self._lock = threading.Lock()
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.allocated_budget = 0.0
        # Track per-instance usage
        self.instance_allocations: Dict[str, float] = {}

    def request_budget(self, instance_id: str, amount: float) -> float:
        """
        Atomically request a budget chunk.
        Returns the amount granted (may be less than requested or 0).
        """
        with self._lock:
            if self.remaining_budget <= 0:
                return 0.0
            
            grant = min(amount, self.remaining_budget)
            self.remaining_budget -= grant
            self.allocated_budget += grant
            
            current = self.instance_allocations.get(instance_id, 0.0)
            self.instance_allocations[instance_id] = current + grant
            
            return grant

    def return_unused(self, instance_id: str, amount: float):
        """
        Return unused budget (e.g., end of window or shutdown).
        """
        with self._lock:
            self.remaining_budget += amount
            self.allocated_budget -= amount
            if instance_id in self.instance_allocations:
                self.instance_allocations[instance_id] -= amount

    def get_global_state(self):
        with self._lock:
            return {
                "total": self.total_budget,
                "remaining": self.remaining_budget,
                "allocated": self.allocated_budget,
                "percent_spent": 1.0 - (self.remaining_budget / self.total_budget)
            }
