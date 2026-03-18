from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

MethodName = Literal["waiting_time_mc", "pps_mc", "procedure_a", "procedure_b"]
BackendName = Literal["exact", "gaussian"]


@dataclass(frozen=True)
class Tolerances:
    rtol: float = 1e-9
    atol: float = 1e-12


@dataclass
class ModelSpec:
    L: int
    H: Optional[np.ndarray]
    jump_ops: List[np.ndarray]
    gamma: float
    w: float = 0.0
    boundary: str = "open"
    model_id: str = "generic"
    initial_state: Optional[np.ndarray] = None
    initial_gamma: Optional[np.ndarray] = None
    # Gaussian backend fields
    majorana_A_eff: Optional[np.ndarray] = None
    majorana_jump_pairs: Optional[List[tuple[int, int]]] = None
    initial_V: Optional[np.ndarray] = None


@dataclass
class SimulationConfig:
    T: float
    zeta: float
    n_traj: int
    seed: int = 0
    backend: BackendName = "exact"
    method: MethodName = "pps_mc"
    tolerances: Tolerances = field(default_factory=Tolerances)


@dataclass
class TrajectoryRecord:
    times: List[float]
    accepted_jump_times: List[float]
    candidate_jump_times: List[float]
    channels: List[int]
    n_clicks: int
    accepted: bool
    observables: Dict[str, Any]
    # debug fields
    final_time: float
    candidate_count: int
