from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StateBackend(ABC):
    @abstractmethod
    def copy(self) -> "StateBackend":
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def purity(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def propagate_no_click(self, tau: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def survival(self, tau: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def channel_rates(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def apply_jump(self, j: int) -> None:
        raise NotImplementedError

    def analytic_waiting_time(self, r: float) -> float | None:
        return None
