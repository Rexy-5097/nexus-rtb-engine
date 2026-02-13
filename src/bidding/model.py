import os
import logging
import numpy as np
import pickle # Kept only for legacy fallback with signature
from typing import Dict, Any, Optional, Tuple

from src.bidding.config import config
from src.utils.crypto import ModelIntegrity

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Manages the lifecycle, security, and loading of machine learning model artifacts.
    
    Strategies:
        1. NumPy (.npz): Preferred. Fast, secure, zero-code-execution risk.
        2. Pickle (.pkl): Legacy. Requires SHA256 signature.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the loader.

        Args:
            model_path (str): Absolute path to the model file (.npz or .pkl).
        """
        self.model_path = model_path
        
        # State: Model Parameters
        self.weights_ctr: Optional[np.ndarray] = None
        self.weights_cvr: Optional[np.ndarray] = None
        self.intercept_ctr: float = 0.0
        self.intercept_cvr: float = 0.0
        self.stats: Dict[str, Any] = {}
        
        # FAIL-CLOSED FLAG
        self.model_loaded = False
        
        # Immediate load attempts
        self._load_safe()

    def _load_safe(self) -> None:
        """
        Attempt to load the model with strict security and validation checks.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model artifact not found: {self.model_path}. Engine DISABLED.")
            self.model_loaded = False
            return

        # Strategy 1: NumPy (.npz) - SAFE
        if self.model_path.endswith(".npz"):
            self._load_numpy()
            return

        # Strategy 2: Pickle (.pkl) - RISKY (Requires Signature)
        if self.model_path.endswith(".pkl"):
            self._load_pickle_secure()
            return
            
        logger.error(f"Unknown model format: {self.model_path}")
        self.model_loaded = False

    def _load_numpy(self):
        """Load from safe .npz format."""
        try:
            with np.load(self.model_path, allow_pickle=False) as data:
                # CTR
                if "ctr_coef" in data and "ctr_intercept" in data:
                    ctr_coef = data["ctr_coef"]
                    if self._validate_shape(ctr_coef, config.model.hash_space):
                        self.weights_ctr = ctr_coef
                        self.intercept_ctr = float(data["ctr_intercept"])
                
                # CVR
                if "cvr_coef" in data and "cvr_intercept" in data:
                    cvr_coef = data["cvr_coef"]
                    if self._validate_shape(cvr_coef, config.model.hash_space):
                        self.weights_cvr = cvr_coef
                        self.intercept_cvr = float(data["cvr_intercept"])
                        
                # If we got here, we are good? 
                # Strict check: Must have both? Or at least CTR?
                if self.weights_ctr is not None:
                     self.model_loaded = True
                     logger.info("Model loaded from .npz successfully.")
                else:
                     logger.error("Model .npz missing critical weights.")
                     self.model_loaded = False

        except Exception as e:
            logger.critical(f"Failed to load .npz model: {e}", exc_info=True)
            self.model_loaded = False

    def _load_pickle_secure(self):
        """Load legacy pickle with signature enforcement."""
        sig_path = f"{self.model_path}.sig"
        if not ModelIntegrity.verify_signature(self.model_path, sig_path):
            logger.critical("SECURITY ALERT: Model signature verification failed. Refusing to load.")
            self.model_loaded = False
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            if self._parse_and_validate(data):
                self.model_loaded = True
                logger.info("Legacy model loaded from .pkl (Verified).")
            else:
                self.model_loaded = False
        except Exception as e:
            logger.critical(f"Fatal error loading legacy model: {e}", exc_info=True)
            self.model_loaded = False

    def _parse_and_validate(self, data: Dict[str, Any]) -> bool:
        """Validate tensor shapes from dict."""
        expected_dim = config.model.hash_space
        valid_ctr = False

        # CTR Model
        if "ctr" in data:
            coef = data["ctr"].get("coef")
            intercept = data["ctr"].get("intercept")
            
            if self._validate_shape(coef, expected_dim):
                self.weights_ctr = coef.flatten()
                self.intercept_ctr = float(intercept[0]) if intercept is not None else 0.0
                valid_ctr = True

        # CVR Model
        if "cvr" in data:
            coef = data["cvr"].get("coef")
            intercept = data["cvr"].get("intercept")
            
            if self._validate_shape(coef, expected_dim):
                self.weights_cvr = coef.flatten()
                self.intercept_cvr = float(intercept[0]) if intercept is not None else 0.0
                
        # Metadata
        if "stats" in data:
            self.stats = data["stats"]
            
        return valid_ctr

    @staticmethod
    def _validate_shape(coef: Any, expected_dim: int) -> bool:
        if coef is None: return False
        try:
            shape = coef.shape
            if len(shape) == 1 and shape[0] == expected_dim: return True
            if len(shape) == 2 and shape[1] == expected_dim: return True
            return False
        except Exception:
            return False

    def get_stats(self, advertiser_id: str) -> Dict[str, float]:
        return self.stats.get(str(advertiser_id), {"avg_mp": 50.0, "avg_ev": 0.001})

    @staticmethod
    def _validate_shape(coef: Any, expected_dim: int) -> bool:
        if coef is None: return False
        try:
            shape = coef.shape
            if len(shape) == 1 and shape[0] == expected_dim: return True
            if len(shape) == 2 and shape[1] == expected_dim: return True
            return False
        except Exception:
            return False

    def _reset_defaults(self) -> None:
        """Revert to fail-safe default intercepts."""
        self.weights_ctr = None
        self.weights_cvr = None
        self.intercept_ctr = config.model.default_intercept_ctr
        self.intercept_cvr = config.model.default_intercept_cvr

    def get_stats(self, advertiser_id: str) -> Dict[str, float]:
        return self.stats.get(str(advertiser_id), {"avg_mp": 50.0, "avg_ev": 0.001})
