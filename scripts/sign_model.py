#!/usr/bin/env python3
import sys
import os
import argparse
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.crypto import ModelIntegrity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignModel")

def main():
    parser = argparse.ArgumentParser(description="Sign a model file (generate SHA256 hash)")
    parser.add_argument("model_path", help="Path to the model file (.pkl)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"File not found: {args.model_path}")
        sys.exit(1)
        
    logger.info(f"Computing hash for {args.model_path}...")
    file_hash = ModelIntegrity.compute_hash(args.model_path)
    
    sig_path = args.model_path + ".sig"
    with open(sig_path, "w") as f:
        f.write(file_hash)
        
    logger.info(f"Signature written to {sig_path}")
    logger.info(f"Hash: {file_hash}")

if __name__ == "__main__":
    main()
