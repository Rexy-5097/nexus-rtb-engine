# Security Policy

## Threat Model

We assume the Nexus-RTB engine operates in a hostile environment where:

1.  **External Input (Bid Requests)** is untrusted and potentially malicious.
2.  **Model Artifacts** could be tampered with if the S3 bucket/storage is compromised.
3.  **Budget Control** is critical; failure leads to financial loss.

## Vulnerability Mitigation

| Threat             | Surface            | Mitigation Strategy                                                                                                                                 |
| ------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RCE via Pickle** | `ModelLoader`      | **Safe Loading via NumPy**. The engine defaults to `.npz` files using `allow_pickle=False`. Legacy `.pkl` files require a valid `SHA256` signature. |
| **DoS (Memory)**   | `BidRequest`       | **Input Truncation**. All string inputs (UA, URL) are truncated to 512 chars before hashing. Strict Pydantic types enforce schema.                  |
| **Budget Drain**   | `PacingController` | **Atomic Circuit Breakers**. Hard constraints on Daily ($25M), Hourly ($2M), and Minute ($50k) spend. Fails closed if limits are hit.               |
| **Data Poisoning** | `Training`         | **Outlier detection**. Updates with erratic gradients are discarded.                                                                                |

## Secure Development Lifecycle (SDLC)

1.  **Dependencies**: All packages pinned in `requirements.txt`.
2.  **Secrets**: No hardcoded API keys. Configuration via `os.environ` only.
3.  **Images**: Distroless or Slim Docker images to reduce attack surface.
4.  **Financial Safety**: Logic gates prevent bidding if ROI < 0 or if win rate deviates significantly (`> 50%`).

## Reporting a Vulnerability

Please report security issues to `security-team@nexus-rtb.internal` (Mock).
**DO NOT** create public GitHub issues for security vulnerabilities.

---

_Last Audit: February 2026_
