import hashlib
import hmac
import logging
import os

logger = logging.getLogger(__name__)


class ModelIntegrity:
    """
    Utilities for model signature verification to prevent tampering.
    """

    @staticmethod
    def compute_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def verify_signature(model_path: str, sig_path: str) -> bool:
        """
        Verify that the model file matches the detached signature (hash).
        In a real scenario, this would check a cryptographic signature signed by a private key.
        Here we verify the SHA256 hash against the published .sig file.
        """
        if not os.path.exists(model_path) or not os.path.exists(sig_path):
            logger.error("Model or signature file missing.")
            return False

        try:
            computed_hash = ModelIntegrity.compute_hash(model_path)
            with open(sig_path, "r") as f:
                stored_hash = f.read().strip()

            # Constant time comparison to avoid timing attacks (though strictly more relevant for HMAC)
            if hmac.compare_digest(computed_hash, stored_hash):
                return True
            else:
                logger.critical(
                    f"Hash mismatch! Computed: {computed_hash}, Expected: {stored_hash}"
                )
                return False
        except Exception as e:
            logger.error(f"Error validating signature: {e}")
            return False
