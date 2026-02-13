import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional

from src.bidding.config import config
from src.utils.crypto import ModelIntegrity

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles safe loading of model weights and statistics.
    Implements fail-safe fallbacks if the model file is missing or corrupted.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sig_path = model_path + ".sig"
        
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
        enforces integrity check via SHA256 checksum.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}. Using fail-safe defaults.")
            return

        # --- SECURITY: Signature Verification ---
        if not ModelIntegrity.verify_signature(self.model_path, self.sig_path):
            logger.critical("Model signature verification failed! Possible tampering detected. Refusing to load weights.")
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            
            # --- SECURITY: Schema & Version Validation ---
            # In a real scenario, we'd check data.get("version") compatible with APP_VERSION
            
            # Helper to validate shape
            def validate_shape(coef, expected_features):
                if coef is None: return True
                # Coef shape is (1, HASH_SPACE) or flattened
                if hasattr(coef, "shape"):
                    if len(coef.shape) == 1 and coef.shape[0] == expected_features: return True
                    if len(coef.shape) == 2 and coef.shape[1] == expected_features: return True
                return False

            HASH_SPACE = config.model.hash_space

            if "ctr" in data:
                ctr_coef = data["ctr"]["coef"].flatten()
                if validate_shape(ctr_coef, HASH_SPACE):
                    self.weights_ctr = ctr_coef
                    self.intercept_ctr = float(data["ctr"]["intercept"][0])
                else:
                    logger.error("CTR model shape mismatch. Ignoring.")
            
            if "cvr" in data:
                cvr_coef = data["cvr"]["coef"].flatten()
                if validate_shape(cvr_coef, HASH_SPACE):
                    self.weights_cvr = cvr_coef
                    self.intercept_cvr = float(data["cvr"]["intercept"][0])
                    
            if "stats" in data:
                self.stats = data["stats"]
                
            logger.info("Model loaded successfully and verified.")
            
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
