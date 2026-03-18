from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def spawn_rng(parent: np.random.Generator) -> np.random.Generator:
    seed = int(parent.integers(0, 2**63 - 1, dtype=np.int64))
    return np.random.default_rng(seed)
