import hashlib
import sys
import os

model_path = "src/model_weights.pkl"
sig_path = model_path + ".sig"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    sys.exit(1)

print(f"Computing hash for {model_path}...")
sha256_hash = hashlib.sha256()
with open(model_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
computed_hash = sha256_hash.hexdigest()

print(f"Writing signature to {sig_path}...")
with open(sig_path, "w") as f:
    f.write(computed_hash)
    
print("Done.")
