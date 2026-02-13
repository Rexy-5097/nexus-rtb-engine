import zlib
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Hasher:
    """
    High-performance hashing utilities for the "Hashing Trick".
    
    Strategies:
        - Adler32: Extremely fast, non-cryptographic. Good for high-throughput inference.
        - SHA256: Cryptographic, slower. Good for ID generation or security.
    """
    
    @staticmethod
    def adler32_hash(s: Optional[str], hash_space: int) -> int:
        """
        Compute a fast hash for feature bucketization.
        
        Args:
            s (str): Input string.
            hash_space (int): Modulo for the hash space (e.g., 2^18).
            
        Returns:
            int: Hashed index.
        """
        if not s:
            return 0
        # zlib.adler32 returns unsigned 32-bit int on Python 3
        # We mask strictly to ensuring consistency across platforms
        return (zlib.adler32(s.encode("utf-8")) & 0xffffffff) % hash_space

    @staticmethod
    def sha256_hash(s: Optional[str], hash_space: int) -> int:
        """
        Compute a cryptographically secure hash.
        """
        if not s:
            return 0
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        # Use first 4 bytes (32-bit int)
        val = int.from_bytes(digest[:4], 'big') 
        return val % hash_space

def hash_feature(value: str, hash_space: int) -> int:
    """
    Standard interface for feature hashing used by the Bidding Engine.
    Defaults to Adler32 for latency performance (<500ns).
    
    Args:
        value (str): Feature string (e.g., "ua:chrome").
        hash_space (int): Size of the feature vector.
        
    Returns:
        int: Index in the feature vector.
    """
    return Hasher.adler32_hash(value, hash_space)
