"""Active PPS/Doob quantum-jump implementation surface."""
from __future__ import annotations

from .types import JumpTrajectory, Tolerances

try:  # Python 3.8+ standard
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    try:
        __version__ = _pkg_version("pps-qj")
    except PackageNotFoundError:  # editable install with mismatched dist
        __version__ = "0.0.0+unknown"
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["JumpTrajectory", "Tolerances", "__version__"]
