"""Makes ``scripts/common`` importable by the test suite.

pytest prepends the directory holding this file to sys.path, so placing it at
the repository root is enough for ``tests/`` to import the shared modules by
name. No PYTHONPATH export and no editable install required.
"""

import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent / "scripts" / "common"
if COMMON.is_dir():
    sys.path.insert(0, str(COMMON))
else:
    # Flat layout: the modules sit next to this file.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
