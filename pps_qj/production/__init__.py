"""Production entry point for the QJ-PPS (partial post-selection quantum-jump)
simulator.

This subpackage is the ONE supported way to run a production Cut B cloning
cell.  Everything in it wraps already-validated components in
``pps_qj.cloning``; no sampler was reimplemented here.

See ``docs/PRODUCTION_ALGORITHM.md`` for the certified feature ledger and for
which optimisations are deliberately excluded.
"""

from pps_qj.production.config import (
    ALGORITHM_VERSION,
    OUTPUT_SCHEMA_VERSION,
    ProductionConfig,
)

__all__ = ["ProductionConfig", "ALGORITHM_VERSION", "OUTPUT_SCHEMA_VERSION"]
