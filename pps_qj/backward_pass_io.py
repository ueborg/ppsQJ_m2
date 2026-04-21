"""Serialisation layer for ``GaussianBackwardData``.

Provides ``save_backward_pass`` and ``load_backward_pass``. The loaded object
is a duck-typed drop-in replacement for ``GaussianBackwardData`` in
``doob_gaussian_trajectory``: it exposes ``state_at(t) -> (C_t, z_t)`` and the
attributes ``T``, ``zeta``, ``Z_T``, ``theta_doob``.

Z_T CONVENTION — read carefully
-------------------------------

The Gaussian backward ODE (see ``backward_pass.gaussian_backward_rhs``)
evolves a pair ``(C(tau), log z(tau))`` in reverse time ``tau = T - t``, with
``y0[-1] = 0`` at ``tau = 0``. Here ``C`` is the covariance of the *normalised*
Gaussian operator ``G(tau) / Tr(G(tau))`` and ``z = 2^{-L} Tr(G)`` is the
scalar prefactor. At ``tau = 0`` the operator is the identity: ``C = 0``,
``z = 1``, so ``log z = 0`` is the correct initial condition.

The partition function is

    Z_T = Tr( G(tau=T) * rho_0 ) = Tr( G_0 * rho_0 )

and from ``gaussian_overlap`` this equals

    Z_T = z(tau=T) * sqrt( det( I - C(tau=T) @ Gamma_0 ) ),

where ``Gamma_0`` is the Majorana covariance of the initial pure state
``rho_0``. It is therefore NOT correct to identify ``Z_T`` with ``z(tau=T)``
alone — the ``sqrt(det(...))`` factor is in general nontrivial, matching how
``doob_wtmc.doob_gaussian_trajectory`` computes the Doob denominator at
``t = 0`` (see ``gaussian_overlap`` call in that file).

We therefore compute ``Z_T`` explicitly via ``gaussian_overlap`` at save time
and store it as a separate scalar, rather than re-deriving it on load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pps_qj.gaussian_backend import neel_covariance, orbitals_from_covariance
from pps_qj.overlaps import gaussian_overlap


__all__ = ["save_backward_pass", "load_backward_pass", "LoadedBackwardPass"]


def _compute_Z_T(sample_tau: np.ndarray,
                 sample_covariances: np.ndarray,
                 sample_z: np.ndarray,
                 L: int) -> float:
    # sample_tau is ascending [0, ..., T]; physical time t = T - tau, so
    # t = 0 corresponds to the LAST sample entry.
    C_t0 = np.asarray(sample_covariances[-1], dtype=np.float64)
    z_t0 = float(sample_z[-1])
    gamma0 = neel_covariance(L)
    return float(gaussian_overlap(C_t0, gamma0, z_scalar=z_t0))


def save_backward_pass(result, path, metadata: dict) -> None:
    """Save a ``GaussianBackwardData`` result to a ``.npz`` file.

    Stored arrays (with ``t_grid`` ordered in ascending physical time):
        t_grid    : shape (n_grid,), t_grid[0] = 0, t_grid[-1] = T
        C_grid    : shape (n_grid, 2L, 2L)
        z_grid    : shape (n_grid,)
        Z_T       : scalar, Tr(G_0 rho_0)
        theta_doob: scalar, log(Z_T) / T

    Metadata keys (expected: L, alpha, w, zeta, T, lam) are saved verbatim as
    0-d arrays alongside the data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tau = np.asarray(result.sample_tau, dtype=np.float64)
    Cs = np.asarray(result.sample_covariances, dtype=np.float64)
    zs = np.asarray(result.sample_z, dtype=np.float64)
    T = float(result.T)

    # Reverse to ascending physical-time order: t = T - tau.
    order = np.argsort(T - tau)
    t_grid = (T - tau)[order]
    C_grid = Cs[order]
    z_grid = zs[order]

    L = int(metadata["L"])
    Z_T = _compute_Z_T(tau, Cs, zs, L)
    theta_doob = float(np.log(max(Z_T, 1e-300)) / T) if T > 0 else 0.0

    to_save = {
        "t_grid": t_grid,
        "C_grid": C_grid,
        "z_grid": z_grid,
        "Z_T": np.asarray(Z_T, dtype=np.float64),
        "theta_doob": np.asarray(theta_doob, dtype=np.float64),
    }
    for k, v in metadata.items():
        to_save[f"meta_{k}"] = np.asarray(v)

    # np.savez auto-appends ".npz" if missing; write to a sibling .tmp.npz
    # then rename atomically onto the target.
    tmp_path = path.with_name(path.name + ".tmp")
    np.savez(str(tmp_path), **to_save)
    # savez has written either tmp_path or tmp_path+'.npz' depending on suffix.
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    written.replace(path)


@dataclass
class LoadedBackwardPass:
    """Drop-in replacement for ``GaussianBackwardData`` for use in Doob WTMC."""

    t_grid: np.ndarray
    C_grid: np.ndarray
    z_grid: np.ndarray
    orbitals_grid: np.ndarray   # (N, 2L, L) complex
    log_z_grid: np.ndarray      # (N,) float
    T: float
    zeta: float
    Z_T: float
    theta_doob: float
    metadata: dict

    def state_at(self, t: float) -> tuple[np.ndarray, float]:
        t = float(t)
        if not (0.0 <= t <= self.T + 1e-12):
            raise ValueError(f"t={t} must lie in [0, T={self.T}]")
        t_clamped = min(max(t, 0.0), self.T)
        idx_right = int(np.searchsorted(self.t_grid, t_clamped, side="left"))
        if idx_right <= 0:
            return self.C_grid[0].copy(), float(self.z_grid[0])
        if idx_right >= len(self.t_grid):
            return self.C_grid[-1].copy(), float(self.z_grid[-1])
        t_lo = self.t_grid[idx_right - 1]
        t_hi = self.t_grid[idx_right]
        if t_hi == t_lo:
            return self.C_grid[idx_right].copy(), float(self.z_grid[idx_right])
        w = (t_clamped - t_lo) / (t_hi - t_lo)
        C = (1.0 - w) * self.C_grid[idx_right - 1] + w * self.C_grid[idx_right]
        z = (1.0 - w) * self.z_grid[idx_right - 1] + w * self.z_grid[idx_right]
        C = 0.5 * (C - C.T)
        return C, float(z)

    def orbitals_at(self, t: float) -> tuple[np.ndarray, float]:
        """Return (backward_orbitals W, log_z) at time t."""
        t = float(t)
        if not (0.0 <= t <= self.T + 1e-12):
            raise ValueError(f"t={t} must lie in [0, T={self.T}]")
        t_clamped = min(max(t, 0.0), self.T)
        idx_right = int(np.searchsorted(self.t_grid, t_clamped, side="left"))
        if idx_right <= 0:
            return self.orbitals_grid[0].copy(), float(self.log_z_grid[0])
        if idx_right >= len(self.t_grid):
            return self.orbitals_grid[-1].copy(), float(self.log_z_grid[-1])
        t_lo = self.t_grid[idx_right - 1]
        t_hi = self.t_grid[idx_right]
        if t_hi == t_lo:
            return self.orbitals_grid[idx_right].copy(), float(self.log_z_grid[idx_right])
        w = (t_clamped - t_lo) / (t_hi - t_lo)
        orbs = (1.0 - w) * self.orbitals_grid[idx_right - 1] + w * self.orbitals_grid[idx_right]
        orbs, _ = np.linalg.qr(orbs, mode="reduced")
        log_z = float((1.0 - w) * self.log_z_grid[idx_right - 1] + w * self.log_z_grid[idx_right])
        return orbs, log_z


def load_backward_pass(path) -> LoadedBackwardPass:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        t_grid = np.asarray(data["t_grid"], dtype=np.float64)
        C_grid = np.asarray(data["C_grid"], dtype=np.float64)
        z_grid = np.asarray(data["z_grid"], dtype=np.float64)
        Z_T = float(data["Z_T"])
        theta_doob = float(data["theta_doob"])
        metadata: dict = {}
        for k in keys:
            if k.startswith("meta_"):
                metadata[k[len("meta_"):]] = data[k].item() if data[k].ndim == 0 else data[k]

    T = float(metadata.get("T", t_grid[-1]))
    zeta = float(metadata.get("zeta", 1.0))

    # Compute orbitals and log_z from stored data — avoids changing save format.
    n = C_grid.shape[1]
    L = n // 2
    orbitals_grid = np.empty((len(t_grid), n, L), dtype=np.complex128)
    log_z_grid = np.where(z_grid > 0, np.log(z_grid), -np.inf)
    for i in range(len(t_grid)):
        orbitals_grid[i] = orbitals_from_covariance(C_grid[i])

    return LoadedBackwardPass(
        t_grid=t_grid,
        C_grid=C_grid,
        z_grid=z_grid,
        orbitals_grid=orbitals_grid,
        log_z_grid=log_z_grid,
        T=T,
        zeta=zeta,
        Z_T=Z_T,
        theta_doob=theta_doob,
        metadata=metadata,
    )
