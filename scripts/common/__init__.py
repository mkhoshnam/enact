"""Shared modules for reproducing the reported experiments.

Importable either as a package (``from scripts.common import paper_protocol``)
or flat, with this directory on ``sys.path``. Scripts elsewhere in the repo can
use the flat form with one line:

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
"""

from . import paper_protocol  # noqa: F401
from . import future_transforms  # noqa: F401
from . import rafc_variants  # noqa: F401
from . import result_io  # noqa: F401

__all__ = [
    "paper_protocol",
    "future_transforms",
    "rafc_variants",
    "result_io",
]
