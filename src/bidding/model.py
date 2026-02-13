import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.bidding.config import config
from src.utils.crypto import ModelIntegrity

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Manages the lifecycle, security, and loading of machine learning model artifacts.
    
    Responsibilities:
        - Authenticate model artifacts (SHA256 signature).
        - Validate schema and tensor shapes.
        - Provide fail-safe fallback values on corruption or missing files.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the loader.

        Args:
            model_path (str): Absolute path to the .pkl model file.
        """
        self.model_path = model_path
        self.sig_path = f"{model_path}.sig"
        
        # State: Model Parameters (Default: Safe Intercepts)
        self.weights_ctr: Optional[np.ndarray] = None
        self.weights_cvr: Optional[np.ndarray] = None
        self.intercept_ctr: float = config.model.default_intercept_ctr
        self.intercept_cvr: float = config.model.default_intercept_cvr
        self.stats: Dict[str, Any] = {}
        
        # Immediate load attempts
        self._load_safe()

    def _load_safe(self) -> None:
        """
        Attempt to load the model with strict security and validation checks.
        Failures result in a logged critical error and fallback to default (safe) parameters.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model artifact not found: {self.model_path}. Running in Fail-Safe mode.")
            return

        # 1. Integrity Check
        if not ModelIntegrity.verify_signature(self.model_path, self.sig_path):
            logger.critical("SECURITY ALERT: Model signature verification failed. Possible tampering. Refusing to load.")
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            
            # 2. Schema Validation
            self._parse_and_validate(data)
            logger.info("Model loaded and verified successfully.")
            
        except Exception as e:
            logger.critical(f"Fatal error loading model: {e}", exc_info=True)
            self._reset_defaults()

    def _parse_and_validate(self, data: Dict[str, Any]) -> None:
        """
        Validate tensor shapes and extract weights.
        
        Args:
            data (dict): The unpickled model dictionary.
        """
        expected_dim = config.model.hash_space

        # CTR Model
        if "ctr" in data:
            coef = data["ctr"].get("coef")
            intercept = data["ctr"].get("intercept")
            
            if self._validate_shape(coef, expected_dim):
                self.weights_ctr = coef.flatten()
                self.intercept_ctr = float(intercept[0]) if intercept is not None else -4.0
            else:
                logger.error("CTR weight shape mismatch.")

        # CVR Model
        if "cvr" in data:
            coef = data["cvr"].get("coef")
            intercept = data["cvr"].get("intercept")
            
            if self._validate_shape(coef, expected_dim):
                self.weights_cvr = coef.flatten()
                self.intercept_cvr = float(intercept[0]) if intercept is not None else -4.0
            else:
                logger.error("CVR weight shape mismatch.")
                
        # Metadata
        if "stats" in data:
            self.stats = data["stats"]

    @staticmethod
    def _validate_shape(coef: Any, expected_dim: int) -> bool:
        """
        Verify that the coefficient vector matches the configured hash space.
        
        Args:
            coef (np.ndarray): The weight matrix.
            expected_dim (int): Required number of features.
        
        Returns:
            bool: True if valid.
        """
        if coef is None:
            return False
            
        try:
            # Handle sklearn's (1, N) or flattened (N,) shapes
            if hasattr(coef, "shape"):
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
        """
        Get market statistics for a specific advertiser.
        
        Args:
            advertiser_id (str): The advertiser ID.
            
        Returns:
            Dict: Key metrics like 'avg_mp' (market price).
        """
        return self.stats.get(str(advertiser_id), {"avg_mp": 70.0, "avg_ev": 0.001})
