import pickle
import sys
import os

# Create src directory if not exists (it should)
os.makedirs("src", exist_ok=True)

backup_path = "model_backup.pkl"
output_path = "src/model_weights.pkl"

if not os.path.exists(backup_path):
    print(f"Error: {backup_path} not found.")
    # Create a dummy one if backup missing?
    # Better to fail if no model at all.
    sys.exit(1)

print(f"Loading {backup_path}...")
with open(backup_path, "rb") as f:
    data = pickle.load(f)

print("Keys in backup:", data.keys())

# Add new fields if missing
if "scaler" not in data:
    print("Adding empty scaler...")
    data["scaler"] = {} # Empty scaler means features used raw.

if "calibration" not in data:
    print("Adding default calibration...")
    data["calibration"] = {"a": 1.0, "b": 0.0}

if "adv_priors" not in data:
    print("Adding empty priors...")
    data["adv_priors"] = {}

if "n_map" not in data:
    print("Adding empty N map...")
    data["n_map"] = {}

print(f"Saving to {output_path}...")
with open(output_path, "wb") as f:
    pickle.dump(data, f)
    
print("Done.")
