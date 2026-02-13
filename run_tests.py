import sys
import os
import pytest

# Add project root to sys.path
sys.path.append(os.getcwd())

# Run pytest programmatically
exit_code = pytest.main(["tests/test_features.py", "tests/test_engine.py", "-v"])
sys.exit(exit_code)
