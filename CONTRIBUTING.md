# Contributing to Nexus-RTB

We welcome contributions to the Nexus-RTB engine. As a high-performance system with strict financial safety constraints, we enforce specific engineering standards.

## Development Standards

### 1. Code Quality

- **Type Hints**: All code must have full type annotations (`mypy` strict mode compatibility).
- **Docstrings**: Public methods must use Google-style docstrings.
- **Formatting**: We use `black` for code formatting.

### 2. Safety & Security

- **Fail-Closed**: All error paths must result in a "No Bid" (`bidPrice=0`) response. Never swallow exceptions silently.
- **No Pickles**: Do not use `pickle` for untrusted data. Use `numpy.load` or JSON.
- **Budget Checks**: Never bypass the `PacingController`.

### 3. Testing

- **Unit Tests**: Must cover positive, negative, and boundary cases.
- **Regression**: Ensure `pytest tests/` passes 100% before submitting.
- **Benchmarks**: If you modify the critical path (`process()` method), you must verify latency remains < 5ms.

## Pull Request Process

1.  Create a feature branch (`feat/your-feature`) or bugfix branch (`fix/issue-id`).
2.  Commit changes with conventional commit messages (e.g., `feat: add robust scaler`).
3.  Ensure CI passes (Lint + Tests).
4.  Request review from the maintainers.

## Release Process

We use Semantic Versioning. Major versions indicate changes to the bidding logic or safety mechanisms.

---

_By contributing, you agree that your code will be licensed under the MIT License._
