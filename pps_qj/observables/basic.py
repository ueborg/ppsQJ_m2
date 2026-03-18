from __future__ import annotations

import numpy as np

from pps_qj.types import TrajectoryRecord


def click_count_hist(records: list[TrajectoryRecord], bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([r.n_clicks for r in records], dtype=int)
    if bins is None:
        bins = max(int(counts.max(initial=0)) + 1, 1)
    hist, edges = np.histogram(counts, bins=bins, range=(0, bins))
    return hist, edges


def waiting_time_hist(records: list[TrajectoryRecord], bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    waits: list[float] = []
    for r in records:
        if len(r.accepted_jump_times) <= 1:
            continue
        arr = np.diff(np.array(r.accepted_jump_times, dtype=float))
        waits.extend(arr.tolist())
    if not waits:
        return np.zeros(bins, dtype=int), np.linspace(0.0, 1.0, bins + 1)
    return np.histogram(np.array(waits), bins=bins)


def acceptance_fraction(records: list[TrajectoryRecord]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if r.accepted else 0.0 for r in records]))


def average_purity(records: list[TrajectoryRecord]) -> float:
    vals = []
    for r in records:
        tr = r.observables.get("purity_trace", [])
        if tr:
            vals.append(float(tr[-1]))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def entanglement_entropy_statevector(psi: np.ndarray, L: int, l_sub: int) -> float:
    if l_sub <= 0 or l_sub >= L:
        return 0.0
    dim_a = 2**l_sub
    dim_b = 2 ** (L - l_sub)
    x = psi.reshape((dim_a, dim_b))
    s = np.linalg.svd(x, compute_uv=False)
    p = np.clip((s**2).real, 1e-15, 1.0)
    return float(-np.sum(p * np.log2(p)))


def entanglement_entropy_gamma(Gamma: np.ndarray, l_sub: int) -> float:
    if l_sub <= 0:
        return 0.0
    n = 2 * l_sub
    G = np.asarray(Gamma[:n, :n], dtype=np.float64)
    eig = np.linalg.eigvals(1j * G)
    # Eigenvalues of iΓ_A come in ±ν_m pairs (real, since iΓ_A is Hermitian).
    # Sorted |eigenvalues| = [ν_1, ν_1, ν_2, ν_2, ...]; take every other to get
    # one representative per pair.
    vals = np.sort(np.abs(np.real(eig)))
    nu = vals[::2]
    x1 = np.clip((1.0 + nu) * 0.5, 1e-15, 1.0)
    x2 = np.clip((1.0 - nu) * 0.5, 1e-15, 1.0)
    return float(-np.sum(x1 * np.log2(x1) + x2 * np.log2(x2)))
