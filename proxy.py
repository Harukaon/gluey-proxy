"""Compatibility entrypoint.

The production module is `claude_proxy.py`; this file keeps older
`uvicorn proxy:app` commands working.
"""

from claude_proxy import app
