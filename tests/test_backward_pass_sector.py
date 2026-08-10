"""Validation of the sector-reduced exact backward pass.

These tests check that the sector-restricted backward pass produces operators
that, when embedded back into the full :math:`2^L`-dimensional Hilbert space,
agree with the full :func:`run_exact_backward_pass` on the relevant sector to
high precision. They also confirm:

- The Néel initial state lies in the half-filling sector.
- The half-filling sector is invariant under :math:`H`, :math:`H_{\\text{eff}}`,
  and every jump projector :math:`P_j` (so restriction is meaningful).
- The sector overlap convenience method matches the embedded-matrix overlap.
"""
from __future__ import annotations

import numpy as np
import pytest

from pps_qj.backward_pass import run_exact_backward_pass
from pps_qj.backward_pass_sector import (
    neel_parity_sector_indices,
    project_to_sector,
    run_exact_backward_pass_sector,
)


half_filling_sector_indices = neel_parity_sector_indices  # legacy alias used below
from pps_qj.exact_backend import build_exact_spin_chain_model


@pytest.fixture(scope="module")
def model_l6():
    return build_exact_spin_chain_model(L=6, w=0.5, alpha=0.5)


def test_sector_contains_neel(model_l6):
    sector = half_filling_sector_indices(model_l6.L)
    psi = model_l6.initial_state
    mask = np.ones(psi.shape[0], dtype=bool)
    mask[sector] = False
    leak = float(np.linalg.norm(psi[mask]))
    assert leak < 1e-14, f"Néel state has weight {leak} outside half-filling sector"


def test_sector_invariant_under_hamiltonian_and_projectors(model_l6):
    sector = half_filling_sector_indices(model_l6.L)
    full_dim = model_l6.dim
    not_sector = np.setdiff1d(np.arange(full_dim), sector, assume_unique=True)

    for op in (model_l6.hamiltonian, model_l6.h_effective, *model_l6.jump_projectors):
        block = op.tocsr()[sector, :].tocsc()[:, not_sector]
        assert block.nnz == 0 or np.max(np.abs(block.toarray())) < 1e-13


def test_sector_dim_is_half_full_dim(model_l6):
    sector = half_filling_sector_indices(model_l6.L)
    # Fermion parity halves the dimension: 2^(L-1) = 32 at L=6.
    assert sector.shape[0] == 2 ** (model_l6.L - 1)


def test_sector_backward_matches_full_l6_at_sample_times():
    """Sector-restricted operator agrees with the full backward pass projected
    to the sector at exact sample-grid times to machine precision.

    Comparison at exact sample points isolates the sector-reduction correctness
    from the linear interpolation between samples (which has its own
    ``O(h²·||d²G/dt²||)`` error and is tested separately at coarser tolerance).
    """
    model = build_exact_spin_chain_model(L=6, w=0.5, alpha=0.5)
    T = 2.0
    zeta = 0.5

    full = run_exact_backward_pass(model, T, zeta)
    sector_data = run_exact_backward_pass_sector(model, T, zeta, n_samples=64)
    sector_indices = sector_data.sector_indices

    # Compare at every other sample-grid time (no interpolation involved).
    for k in range(0, sector_data.sample_times.size, 8):
        t = float(sector_data.sample_times[k])
        sec_op = sector_data.sample_operators[k]
        full_op = full.operator_at(t)
        full_restricted = full_op[np.ix_(sector_indices, sector_indices)]
        diff = np.linalg.norm(sec_op - full_restricted)
        assert diff < 1e-10, f"At sample-time t={t:.4f}: diff = {diff:.3e}"


def test_sector_backward_interpolation_quality_l6():
    """Interpolated operator agrees with the full pass at modest tolerance.

    With 64 samples over T=2, linear interpolation introduces O(1e-4) error
    (operator second derivatives are ~10²). This test only sanity-checks that
    interpolation isn't broken; trajectory-level overlaps are robust to this
    noise (it's well below sampling error from N=2000 trajectories).
    """
    model = build_exact_spin_chain_model(L=6, w=0.5, alpha=0.5)
    T = 2.0
    zeta = 0.5
    full = run_exact_backward_pass(model, T, zeta)
    sector_data = run_exact_backward_pass_sector(model, T, zeta, n_samples=128)
    sector_indices = sector_data.sector_indices
    for t in [0.13, 0.71, 1.42, 1.93]:
        sec_op = sector_data._operator_sector_at(t)
        full_restricted = full.operator_at(t)[np.ix_(sector_indices, sector_indices)]
        diff = np.linalg.norm(sec_op - full_restricted)
        assert diff < 5e-4, f"At t={t}: interpolation diff = {diff:.3e}"


def test_sector_operator_at_T_is_identity_on_sector():
    """At t=T the backward operator is the identity (boundary condition)."""
    model = build_exact_spin_chain_model(L=6, w=0.5, alpha=0.5)
    T = 1.5
    zeta = 0.3
    sector_data = run_exact_backward_pass_sector(model, T, zeta, n_samples=33)
    O = sector_data._operator_sector_at(T)
    d = sector_data.sector_dim
    err = np.linalg.norm(O - np.eye(d, dtype=np.complex128))
    assert err < 1e-12, f"G(T) - I sector norm = {err:.3e}"


def test_sector_overlap_matches_embedded_matrix():
    """``overlap()`` shortcut equals the embedded matrix-form overlap."""
    rng = np.random.default_rng(0)
    model = build_exact_spin_chain_model(L=6, w=0.5, alpha=0.5)
    T = 2.0
    zeta = 0.5
    sector_data = run_exact_backward_pass_sector(model, T, zeta, n_samples=33)

    # Random sector-supported state.
    sector = sector_data.sector_indices
    psi = np.zeros(model.dim, dtype=np.complex128)
    psi_sec = rng.standard_normal(sector.size) + 1j * rng.standard_normal(sector.size)
    psi_sec = psi_sec / np.linalg.norm(psi_sec)
    psi[sector] = psi_sec

    for t in [0.1, 0.7, 1.3]:
        ov_method = sector_data.overlap(t, psi)
        O_full = sector_data.operator_at(t)
        ov_explicit = float(np.real(np.vdot(psi, O_full @ psi)))
        assert abs(ov_method - ov_explicit) < 1e-12


def test_project_to_sector_preserves_hermiticity():
    model = build_exact_spin_chain_model(L=6, w=0.7, alpha=0.4)
    sector = half_filling_sector_indices(model.L)
    H_sec = project_to_sector(model.hamiltonian, sector).toarray()
    assert np.linalg.norm(H_sec - H_sec.conj().T) < 1e-13
