"""Shared test settings."""

import os

# In MCP mode a failed call must fail the test, not fall back in-process unseen.
os.environ.setdefault("DEIXIS_FALLBACK", "none")
