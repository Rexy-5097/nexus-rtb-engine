# Model Integrity Policy

## Overview

To prevent malicious model tampering (e.g., changing weights to bid on arbitrage sites or injecting RCE payloads via pickle), the Nexus-RTB engine enforces strict integrity checks.

## Mechanism

1. **Hashing**: Every valid `model_weights.pkl` has a corresponding `model_weights.pkl.sig` file containing its SHA256 hash.
2. **Verification**: On startup, the engine computes the SHA256 hash of the model file and compares it to the `.sig` file.
3. **Fail-Closed**: If the hashes do not match, the engine **refuses to load the weights** and falls back to safe default intercepts (bidding conservatively).

## Workflow for Data Scientists

### Signing a New Model

When releasing a new model version:

1. Train the model (`train.py` outputs `src/model_weights.pkl`).
2. Run the signing script:
   ```bash
   python scripts/sign_model.py src/model_weights.pkl
   ```
3. Commit both `.pkl` and `.pkl.sig` to the repository / artifact store.

## Threat Model

| Threat                    | Mitigation                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Man-in-the-Middle**     | Integrity check ensures file was not modified in transit (assuming `.sig` is trusted).                                                                 |
| **Disk Corruption**       | Integrity check detects bit rot or truncation.                                                                                                         |
| **Malicious Replacement** | Attacker must update both `.pkl` and `.sig`. In a real production system, `.sig` should be a cryptographic signature signed by an offline private key. |
