"""Garbage detection package: detector, evidence, history, alerts, UI."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import config` from src modules even when imported from other cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
