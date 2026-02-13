import zlib
import hashlib
from typing import Optional

class Hasher:
    """
    Provides consistent hashing utilities for feature extraction.
    Currently uses Adler32 for speed but supports cryptographic hashes for security.
    """
    
    @staticmethod
    def adler32_hash(s: Optional[str], hash_space: int) -> int:
        """
        Fast, non-cryptographic hash for high-throughput feature extraction.
        WARNING: Vulnerable to collision attacks. Use exclusively for feature bucketization.
        """
        if not s:
            return 0
        return (zlib.adler32(s.encode("utf-8")) & 0xffffffff) % hash_space

    @staticmethod
    def sha256_hash(s: Optional[str], hash_space: int) -> int:
        """
        Secure hash for collision-resistant feature extraction.
        Slower than Adler32 but resistant to adversarial attacks.
        """
        if not s:
            return 0
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        # Use first 4 bytes for integer conversion
        return int.from_bytes(digest[:4], 'big') % hash_space
