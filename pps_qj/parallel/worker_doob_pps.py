"""Doob-WTMC worker for a single (L, lambda, zeta) grid point.

Entry point::

    python -m pps_qj.parallel.worker_doob_pps <task_id> <output_dir> [n_workers] [--save-bwd]

For each task this worker:

  1. Builds the ``GaussianChainModel``.
  2. At zeta < 1: runs a backward pass (``run_gaussian_backward_pass``). Saves
     the full pass only for a representative subset of points (selective rule,
     see ``_should_save_full_backward_pass``). Z_T and theta_Doob are always
     captured on the summary regardless.
     At zeta == 1: no backward pass is needed; trajectories use the Born-rule
     sampler and Z_T = 1, theta_Doob = 0.
  3. Runs ``n_traj`` Doob trajectories across ``n_workers`` multiprocessing
     workers (Born-rule when zeta == 1), with deterministic per-worker seeds.
  4. Computes all entanglement observables per trajectory via
     ``compute_all_observables``.
  5. Writes ``doob_{task_id:05d}.npz`` (full stats + per-trajectory arrays)
     and ``summary_{task_id:05d}.json`` (scalar monitoring data, atomic).

SEED SCHEME
-----------
Grid seed ``s`` is deterministic in (L, lam, zeta). Per-worker seed for chunk
index ``i`` is ``s * 100_000 + i``; the RNG for that chunk is seeded from it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from pps_qj.backward_pass import run_gaussian_backward_pass
from pps_qj.backward_pass_io import load_backward_pass, save_backward_pass
from pps_qj.doob_wtmc import doob_gaussian_trajectory
from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    covariance_from_orbitals,
    gaussian_born_rule_trajectory,
)
from pps_qj.observables.topological import compute_all_observables
from pps_qj.parallel.grid_pps import task_params_doob


# Selective backward-pass saving: saves at most 5*3*4 = 60 full passes.
SAVE_BWD_FULL_FOR_L = {16, 32, 64, 128, 256}
SAVE_BWD_FULL_FOR_LAMBDA = {0.3, 0.5, 0.7}
SAVE_BWD_FULL_FOR_ZETA = {0.3, 0.5, 0.7, 1.0}


def _should_save_full_backward_pass(L: int, lam: float, zeta: float) -> bool:
    if L not in SAVE_BWD_FULL_FOR_L:
        return False
    if not any(abs(lam - x) < 0.01 for x in SAVE_BWD_FULL_FOR_LAMBDA):
        return False
    if not any(abs(zeta - z) < 0.01 for z in SAVE_BWD_FULL_FOR_ZETA):
        return False
    return True


def _bwd_file_path(output_dir: Path, L: int, lam: float, zeta: float, T: float) -> Path:
    return output_dir / "backward_passes" / (
        f"bwd_L{L:03d}_lam{lam:.4f}_zeta{zeta:.3f}_T{T:.1f}.npz"
    )


def _n_grid_for_bwd(L: int) -> int:
    return 100 if L <= 96 else 50


# --------------------------------------------------------------------------
# Worker chunk function — must be top-level (picklable for Pool).
# --------------------------------------------------------------------------
def _run_doob_chunk_unpacked(args: tuple) -> list[dict]:
    """Single-argument wrapper so imap_unordered can unpack chunk_args tuples."""
    return _run_doob_chunk(*args)


def _run_doob_chunk(
    L: int,
    w: float,
    alpha: float,
    T: float,
    zeta: float,
    bwd_path_or_none: Optional[str],
    seed_i: int,
    n_chunk: int,
) -> list[dict]:
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng = np.random.default_rng(seed_i)
    jump_pairs_list = list(model.jump_pairs)

    results: list[dict] = []
    if bwd_path_or_none is None:
        # zeta = 1 branch: exact Born rule (no tilt).
        for _ in range(n_chunk):
            traj = gaussian_born_rule_trajectory(model, T=T, rng=rng)
            gamma = np.asarray(traj.final_covariance, dtype=np.float64)
            obs = compute_all_observables(gamma, L, jump_pairs_list)
            obs["n_jumps"] = int(traj.n_jumps)
            results.append(obs)
    else:
        bwd = load_backward_pass(bwd_path_or_none)
        for _ in range(n_chunk):
            traj = doob_gaussian_trajectory(model, bwd, T, zeta, rng)
            # doob_gaussian_trajectory.final_state = orbitals; convert to gamma.
            final_state = np.asarray(traj.final_state)
            if final_state.ndim == 2 and final_state.shape[0] == 2 * L and final_state.shape[1] == L:
                gamma = covariance_from_orbitals(final_state)
            else:
                # Defensive fallback: if it ever becomes a covariance, accept.
                gamma = np.asarray(final_state, dtype=np.float64)
            obs = compute_all_observables(gamma, L, jump_pairs_list)
            obs["n_jumps"] = int(traj.n_jumps)
            results.append(obs)
    return results


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def _nan_stats(values: np.ndarray) -> tuple[float, float, float, int]:
    """(mean, std, err, n_valid) skipping NaNs."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(v))
    std = float(np.std(v))
    err = float(std / np.sqrt(n))
    return mean, std, err, n


def _write_summary_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


def _log(t_start: float, msg: str) -> None:
    """Print a timestamped progress line, flushed immediately."""
    elapsed = time.time() - t_start
    print(f"  [{elapsed:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> tuple[int, Path, int, bool]:
    force_save_bwd = False
    positional: list[str] = []
    for a in argv:
        if a == "--save-bwd":
            force_save_bwd = True
        else:
            positional.append(a)
    if len(positional) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_doob_pps "
            "<task_id> <output_dir> [n_workers] [--save-bwd]"
        )
    task_id = int(positional[0])
    output_dir = Path(positional[1])
    if len(positional) >= 3:
        n_workers = int(positional[2])
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    return task_id, output_dir, max(1, n_workers), force_save_bwd


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    task_id, output_dir, n_workers, force_save_bwd = _parse_args(argv)

    t_start = time.time()

    task = task_params_doob(task_id)
    L = int(task["L"])
    lam = float(task["lam"])
    alpha = float(task["alpha"])
    w = float(task["w"])
    zeta = float(task["zeta"])
    T = float(task["T"])
    n_traj = int(task["n_traj"])
    seed = int(task["seed"])

    print(
        f"\n=== task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}, w={w:.3f}), "
        f"ζ={zeta:.2f}, T={T:.0f}, n_traj={n_traj}, n_workers={n_workers} ===",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backward_passes").mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"doob_{task_id:05d}.npz"
    summary_file = output_dir / f"summary_{task_id:05d}.json"
    bwd_file = _bwd_file_path(output_dir, L, lam, zeta, T)

    # Idempotency guard (on the .npz; the summary JSON is always refreshed).
    if output_file.exists():
        print(f"task {task_id}: already done, skipping ({output_file.name})", flush=True)
        return 0

    try:
        # ------------------- Backward pass --------------------
        Z_T: float
        theta_doob: float
        bwd_path_for_workers: Optional[str]

        if zeta >= 1.0 - 1e-12:
            # Untilted: Z_T = 1, theta = 0, no backward pass needed.
            _log(t_start, "ζ=1 — skipping backward pass (Born-rule branch)")
            Z_T = 1.0
            theta_doob = 0.0
            bwd_path_for_workers = None
        else:
            _log(t_start, f"running backward pass (n_grid={_n_grid_for_bwd(L)}, T={T:.0f}) ...")
            model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
            n_grid = _n_grid_for_bwd(L)
            bwd = run_gaussian_backward_pass(
                model, T=T, zeta=zeta, sample_points=n_grid
            )

            do_persist = force_save_bwd or _should_save_full_backward_pass(L, lam, zeta)
            if do_persist:
                save_backward_pass(
                    bwd,
                    bwd_file,
                    metadata=dict(L=L, alpha=alpha, w=w, zeta=zeta, T=T, lam=lam),
                )
                bwd_path_for_workers = str(bwd_file)
            else:
                # Save to a temp file that workers can read, then delete after.
                tmp_bwd = bwd_file.with_name(bwd_file.name + ".tmp")
                save_backward_pass(
                    bwd,
                    tmp_bwd,
                    metadata=dict(L=L, alpha=alpha, w=w, zeta=zeta, T=T, lam=lam),
                )
                bwd_path_for_workers = str(tmp_bwd)

            # Pull Z_T and theta from the loaded object (single source of truth).
            loaded = load_backward_pass(bwd_path_for_workers)
            Z_T = float(loaded.Z_T)
            theta_doob = float(loaded.theta_doob)
            _log(t_start, f"backward pass done — Z_T={Z_T:.4g}, θ_D={theta_doob:.4f}")

        # ------------------- Trajectories --------------------
        # Split n_traj across n_workers.
        chunk_sizes: list[int] = []
        base = n_traj // n_workers
        rem = n_traj - base * n_workers
        for i in range(n_workers):
            chunk_sizes.append(base + (1 if i < rem else 0))
        chunk_sizes = [c for c in chunk_sizes if c > 0]

        chunk_args = [
            (L, w, alpha, T, zeta, bwd_path_for_workers, seed * 100_000 + i, c)
            for i, c in enumerate(chunk_sizes)
        ]

        all_results: list[dict] = []
        pbar_desc = f"L={L} λ={lam:.2f} ζ={zeta:.2f}"
        if len(chunk_args) == 1 or n_workers == 1:
            _log(t_start, f"running {n_traj} trajectories (single process) ...")
            with tqdm(total=n_traj, desc=pbar_desc, unit="traj") as pbar:
                for args in chunk_args:
                    chunk_result = _run_doob_chunk(*args)
                    all_results.extend(chunk_result)
                    pbar.update(len(chunk_result))
        else:
            import multiprocessing as mp
            n_chunks = len(chunk_args)
            _log(t_start, f"spawning {n_chunks} workers for {n_traj} trajectories ...")
            # fork is safe on Linux and avoids re-importing numpy/scipy per worker,
            # which is critical on CVMFS where imports are slow.
            with mp.get_context("fork").Pool(processes=n_chunks) as pool:
                _log(t_start, "pool ready — trajectories running ...")
                with tqdm(total=n_traj, desc=pbar_desc, unit="traj") as pbar:
                    for chunk_result in pool.imap_unordered(_run_doob_chunk_unpacked, chunk_args):
                        all_results.extend(chunk_result)
                        pbar.update(len(chunk_result))

        # Clean up temp backward pass (if we didn't persist).
        if zeta < 1.0 - 1e-12 and bwd_path_for_workers is not None:
            p = Path(bwd_path_for_workers)
            if p.name.endswith(".npz.tmp"):
                try:
                    p.unlink()
                except OSError:
                    pass

        # ------------------- Statistics --------------------
        def arr_of(key: str) -> np.ndarray:
            return np.array([r[key] for r in all_results], dtype=np.float64)

        S_half_all = arr_of("S_half")
        S_top_all = arr_of("S_top")
        S_top_d_all = arr_of("S_top_d")
        B_L_all = arr_of("B_L")
        B_L_prime_all = arr_of("B_L_prime")
        n_jumps_all = arr_of("n_jumps")

        S_half_mean, S_half_std, S_half_err, _ = _nan_stats(S_half_all)
        S_top_mean, S_top_std, S_top_err, _ = _nan_stats(S_top_all)
        S_top_d_mean, S_top_d_std, S_top_d_err, _ = _nan_stats(S_top_d_all)
        B_L_mean, B_L_std, B_L_err, _ = _nan_stats(B_L_all)
        B_L_prime_mean, B_L_prime_std, B_L_prime_err, _ = _nan_stats(B_L_prime_all)
        n_jumps_mean, n_jumps_std, _, _ = _nan_stats(n_jumps_all)

        wall_time = time.time() - t_start

        # ------------------- Save .npz --------------------
        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, n_traj=n_traj, n_workers=n_workers,
            Z_T=Z_T, theta_doob=theta_doob,
            S_half_mean=S_half_mean, S_half_std=S_half_std, S_half_err=S_half_err,
            S_top_mean=S_top_mean, S_top_std=S_top_std, S_top_err=S_top_err,
            S_top_d_mean=S_top_d_mean, S_top_d_std=S_top_d_std, S_top_d_err=S_top_d_err,
            B_L_mean=B_L_mean, B_L_std=B_L_std, B_L_err=B_L_err,
            B_L_prime_mean=B_L_prime_mean, B_L_prime_std=B_L_prime_std,
            B_L_prime_err=B_L_prime_err,
            n_jumps_mean=n_jumps_mean, n_jumps_std=n_jumps_std,
            S_half_all=S_half_all, B_L_all=B_L_all, B_L_prime_all=B_L_prime_all,
            S_top_all=S_top_all, n_jumps_all=n_jumps_all,
            wall_time=wall_time,
        )

        # ------------------- Summary JSON --------------------
        summary = dict(
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, n_traj=n_traj,
            S_half_mean=S_half_mean, S_half_err=S_half_err,
            S_top_mean=S_top_mean,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            B_L_prime_mean=B_L_prime_mean,
            theta_doob=theta_doob, Z_T=Z_T,
            wall_time=wall_time, status="complete",
        )
        _write_summary_atomic(summary_file, summary)

        print(
            f"task {task_id}: L={L}, λ={lam:.3f}, ζ={zeta:.2f}, T={T:.0f}, "
            f"<S>={S_half_mean:.4f}±{S_half_err:.4f}, "
            f"<B_L>={B_L_mean:.4f}, θ_D={theta_doob:.4f}, t={wall_time:.1f}s",
            flush=True,
        )
        return 0

    except Exception as exc:
        wall_time = time.time() - t_start
        err_summary = dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T,
            status="failed", error=str(exc), wall_time=wall_time,
        )
        try:
            _write_summary_atomic(summary_file, err_summary)
        except Exception:
            pass
        print(f"task {task_id}: FAILED — {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
