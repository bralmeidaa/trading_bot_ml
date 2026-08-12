"""Put the project root on sys.path so research scripts here can import
`backend.*` and `production_trading_system` while living under .claude/research/."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
