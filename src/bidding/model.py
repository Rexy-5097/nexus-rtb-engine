import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional

from src.bidding.config import config

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles safe loading of model weights and statistics.
    Implements fail-safe fallbacks if the model file is missing or corrupted.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weights_ctr = None
        self.weights_cvr = None
        self.intercept_ctr = config.model.default_intercept_ctr
        self.intercept_cvr = config.model.default_intercept_cvr
        self.stats = {}
        
        # Load attempts
        self.load()

    def load(self):
        """
        Load model weights from disk.
        WARNING: Uses pickle.load(). Production systems should verify signatures before loading.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}. Using fail-safe defaults.")
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            
            # Validate structure
            if "ctr" in data:
                self.weights_ctr = data["ctr"]["coef"].flatten()
                self.intercept_ctr = float(data["ctr"]["intercept"][0])
            
            if "cvr" in data:
                self.weights_cvr = data["cvr"]["coef"].flatten()
                self.intercept_cvr = float(data["cvr"]["intercept"][0])
                
            if "stats" in data:
                self.stats = data["stats"]
                
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load model: {e}", exc_info=True)
            # Revert to safe defaults (intercepts only)
            self.weights_ctr = None
            self.weights_cvr = None
            self.intercept_ctr = config.model.default_intercept_ctr
            self.intercept_cvr = config.model.default_intercept_cvr

    def get_stats(self, advertiser_id: str) -> Dict[str, float]:
        """Get market stats for an advertiser."""
        return self.stats.get(str(advertiser_id), {"avg_mp": 70.0, "avg_ev": 0.001})
