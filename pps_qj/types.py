from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Tolerances:
    rtol: float = 1e-9
    atol: float = 1e-12


@dataclass
class JumpTrajectory:
    jump_times: List[float]
    channels: List[int]
    final_time: float
    final_state: Any
    accepted: bool = True
    candidate_jump_times: List[float] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_jumps(self) -> int:
        return len(self.channels)
