# Security Policy

## Threat Model

The Nexus-RTB Engine is designed to operate in a trusted environment (behind a firewall/VPC) but processes untrusted input from Ad Exchanges.

### Trusted Boundaries

- **Internal**: The engine, model weights, and configuration are trusted.
- **External**: Bid requests are untrusted and must be sanitized.

### Known Risks & Mitigations

| Risk                  | Impact       | Mitigation                                                                                                                     |
| --------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Malicious Payload** | RCE / DoS    | Input validation (`src/utils/validation.py`) truncates strings to 512 chars.                                                   |
| **Model Poisoning**   | RCE / Bias   | `model_weights.pkl` is loaded seamlessly, but production should use signed artifacts. Loading is wrapped in try/except blocks. |
| **Replay Attacks**    | Budget Drain | Pacing controller uses a lightweight PID loop. (Note: Distributed replay protection requires Redis).                           |
| **Side-Channel**      | Data Leak    | Error messages are generic. Latency is constant (hashed lookup).                                                               |

## Vulnerability Reporting

Please refer to `docs/security_audit.md` for the initial audit.

If you discover a new vulnerability, please open a private advisory on GitHub.
