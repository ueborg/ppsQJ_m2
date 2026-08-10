#!/usr/bin/env python
"""Exact Q_ζ benchmark driver for a single grid point.

Runs ``N_traj`` trajectories of the exact-Doob or procedure-B sampler at a
specified ``(L, λ, ζ, T)`` grid point, aggregates per-trajectory observables
(half-cut entanglement entropy, click count, final-state purity), and writes
a resumable NumPy ``.npz`` file.

CLI (see ``--help``)::

    python scripts/run_exact_benchmark.py \\
        --L 8 --lambda 0.5 --zeta 0.3 \\
        --T 20.0 --N-traj 2000 \\
        --method exact-doob \\
        --output results/L8/l0.5_z0.3.npz \\
        --n-workers 16 --seed 42

The script sets BLAS thread counts to 1 **before** importing NumPy in each
worker process — running with ``n_workers`` processes each spawning ``L``
BLAS threads produces catastrophic oversubscription on a dense node.
"""
from __future__ import annotations

# BLAS thread pinning — must be set before numpy/scipy import in worker
# processes. In the main process this is a no-op if unset, but the worker
# initialiser below is the authoritative place.
import os as _os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Allow running as ``python scripts/run_exact_benchmark.py`` from any cwd
# by adding the repo root (one level up from this file) to sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


LOG = logging.getLogger("exact_benchmark")


# ------------------------------------------------------------------
# Worker setup — module-level state so the pool doesn't re-import per task.
# ------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _pin_blas_threads() -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        _os.environ[v] = "1"


def _worker_init(config: dict) -> None:
    """Pool initialiser: build the model and backward-pass object once per worker.

    Trajectories are independent, but they share the (deterministic) model and
    backward pass. Building these once saves ~one second per trajectory at L=8.
    """
    _pin_blas_threads()
    # Spawn context workers start without the parent's cwd or sys.path. The
    # parent passes its repo-root path (``pps_qj_path``) so the worker can
    # find the package.
    import sys as _sys
    extra_path = config.get("pps_qj_path")
    if extra_path and extra_path not in _sys.path:
        _sys.path.insert(0, extra_path)
    # Imports inside worker to ensure BLAS pinning takes effect.
    from pps_qj.exact_backend import build_exact_spin_chain_model

    L = int(config["L"])
    lam = float(config["lambda"])
    zeta = float(config["zeta"])
    T = float(config["T"])
    method = str(config["method"])
    backward_method = str(config["backward_method"])

    w = 1.0 - lam
    alpha = lam
    model = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)

    backward_data = None
    if method == "exact-doob":
        if backward_method == "sector":
            from pps_qj.backward_pass_sector import run_exact_backward_pass_sector
            backward_data = run_exact_backward_pass_sector(
                model, T, zeta, n_samples=int(config["backward_samples"])
            )
        elif backward_method == "full":
            from pps_qj.backward_pass import run_exact_backward_pass
            backward_data = run_exact_backward_pass(model, T, zeta)
        else:
            raise ValueError(f"unknown backward_method {backward_method!r}")

    _WORKER_STATE.update(
        model=model,
        backward_data=backward_data,
        T=T,
        zeta=zeta,
        method=method,
        seed=int(config["seed"]),
    )


def _run_one_trajectory(traj_index: int) -> dict:
    """Run one trajectory and return its observables as a dict.

    RNG: seeded deterministically from ``(base_seed, traj_index)`` via
    ``np.random.SeedSequence`` — the seed is stable across runs, so ``--resume``
    produces bit-identical trajectories for already-completed indices.
    """
    from pps_qj.exact_backend import half_chain_entanglement_entropy, procedure_b_trajectory

    state = _WORKER_STATE
    model = state["model"]
    T = state["T"]
    zeta = state["zeta"]
    method = state["method"]
    base_seed = state["seed"]

    seed_seq = np.random.SeedSequence([base_seed, int(traj_index)])
    rng = np.random.default_rng(seed_seq)

    if method == "exact-doob":
        from pps_qj.doob_wtmc import doob_exact_trajectory
        traj = doob_exact_trajectory(model, state["backward_data"], T, zeta, rng)
    elif method == "procedure-b":
        traj = procedure_b_trajectory(model, T, zeta, rng)
    else:
        raise ValueError(f"unknown method {method!r}")

    psi = np.asarray(traj.final_state, dtype=np.complex128)
    entropy = float(half_chain_entanglement_entropy(psi, model.L))
    purity = float(np.abs(np.vdot(psi, psi)))
    return {
        "trajectory_index": int(traj_index),
        "entropy": entropy,
        "n_clicks": int(traj.n_jumps),
        "final_state_purity": purity,
    }


# ------------------------------------------------------------------
# Output I/O — .npz with growing arrays.
# ------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    entropy: np.ndarray
    n_clicks: np.ndarray
    final_state_purity: np.ndarray
    trajectory_index: np.ndarray
    metadata: dict

    @property
    def n_completed(self) -> int:
        return int(self.trajectory_index.shape[0])

    def append(self, records: list[dict]) -> None:
        if not records:
            return
        self.entropy = np.concatenate(
            [self.entropy, np.asarray([r["entropy"] for r in records], dtype=np.float64)]
        )
        self.n_clicks = np.concatenate(
            [self.n_clicks, np.asarray([r["n_clicks"] for r in records], dtype=np.int32)]
        )
        self.final_state_purity = np.concatenate(
            [self.final_state_purity,
             np.asarray([r["final_state_purity"] for r in records], dtype=np.float64)]
        )
        self.trajectory_index = np.concatenate(
            [self.trajectory_index,
             np.asarray([r["trajectory_index"] for r in records], dtype=np.int64)]
        )


def _empty_result(metadata: dict) -> BenchmarkResult:
    return BenchmarkResult(
        entropy=np.empty(0, dtype=np.float64),
        n_clicks=np.empty(0, dtype=np.int32),
        final_state_purity=np.empty(0, dtype=np.float64),
        trajectory_index=np.empty(0, dtype=np.int64),
        metadata=dict(metadata),
    )


def save_result(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a tmp path that already ends in .npz so np.savez doesn't re-append
    # the suffix; atomic rename into place.
    tmp = path.with_name(path.name + ".writing.npz")
    np.savez(
        tmp,
        entropy=result.entropy,
        n_clicks=result.n_clicks,
        final_state_purity=result.final_state_purity,
        trajectory_index=result.trajectory_index,
        metadata=np.asarray(json.dumps(result.metadata)),
    )
    _os.replace(tmp, path)


def load_result(path: Path) -> BenchmarkResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return BenchmarkResult(
            entropy=np.asarray(data["entropy"], dtype=np.float64),
            n_clicks=np.asarray(data["n_clicks"], dtype=np.int32),
            final_state_purity=np.asarray(data["final_state_purity"], dtype=np.float64),
            trajectory_index=np.asarray(data["trajectory_index"], dtype=np.int64),
            metadata=metadata,
        )


# ------------------------------------------------------------------
# Main driver.
# ------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _pps_qj_version() -> str:
    try:
        from pps_qj import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _batched(indices: Iterable[int], size: int) -> Iterable[list[int]]:
    batch: list[int] = []
    for idx in indices:
        batch.append(idx)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    output = Path(args.output)

    import pps_qj as _pps
    pps_qj_path = str(Path(_pps.__file__).resolve().parent.parent)

    config = {
        "L": args.L,
        "lambda": args.lambda_,
        "zeta": args.zeta,
        "T": args.T,
        "method": args.method,
        "backward_method": args.backward_method,
        "backward_samples": args.backward_samples,
        "seed": args.seed,
        "pps_qj_path": pps_qj_path,
    }

    metadata = dict(config)
    metadata.update({
        "N_traj_target": args.N_traj,
        "git_commit": _git_commit(),
        "pps_qj_version": _pps_qj_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    if args.resume and output.exists():
        result = load_result(output)
        # Sanity: the stored metadata should match config except for
        # cumulative fields.
        for key in ("L", "lambda", "zeta", "T", "method", "backward_method"):
            stored = result.metadata.get(key)
            current = metadata[key]
            if stored != current:
                raise RuntimeError(
                    f"--resume: stored {key}={stored!r} != requested {current!r} for {output}"
                )
        LOG.info("resuming from %d completed trajectories", result.n_completed)
    else:
        result = _empty_result(metadata)

    remaining_indices = [
        i for i in range(args.N_traj)
        if i not in set(result.trajectory_index.tolist())
    ]
    if not remaining_indices:
        LOG.info("all %d trajectories already complete — nothing to do", args.N_traj)
        return result

    LOG.info(
        "L=%d λ=%.3f ζ=%.3f method=%s: running %d trajectories on %d workers",
        args.L, args.lambda_, args.zeta, args.method,
        len(remaining_indices), args.n_workers,
    )

    start_walltime = time.perf_counter()
    ctx = mp.get_context("spawn")  # clean BLAS env per worker

    with ctx.Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(config,),
    ) as pool:
        for batch in _batched(remaining_indices, args.checkpoint_every):
            t0 = time.perf_counter()
            records = pool.map(_run_one_trajectory, batch)
            result.append(records)
            # Running walltime stored with every checkpoint.
            result.metadata["walltime_s"] = (
                float(result.metadata.get("walltime_s", 0.0)) +
                (time.perf_counter() - t0)
            )
            result.metadata["N_traj_completed"] = result.n_completed
            save_result(output, result)
            LOG.info(
                "checkpoint: %d/%d trajectories, batch walltime %.1fs, cum %.1fs",
                result.n_completed, args.N_traj,
                time.perf_counter() - t0,
                time.perf_counter() - start_walltime,
            )

    result.metadata["N_traj_completed"] = result.n_completed
    save_result(output, result)
    LOG.info(
        "done: %d trajectories, total walltime %.1fs",
        result.n_completed, time.perf_counter() - start_walltime,
    )
    return result


def _default_n_workers() -> int:
    env = _os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return _os.cpu_count() or 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--lambda", dest="lambda_", type=float, required=True)
    p.add_argument("--zeta", type=float, required=True)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--N-traj", dest="N_traj", type=int, default=2000)
    p.add_argument(
        "--method", choices=("exact-doob", "procedure-b"), default="exact-doob",
    )
    p.add_argument(
        "--backward-method", choices=("sector", "full"), default="sector",
        help="Backward-pass implementation. 'sector' uses the parity-sector reduction; "
             "'full' uses the unreduced 2^L adjoint superoperator (for cross-check at small L).",
    )
    p.add_argument(
        "--backward-samples", type=int, default=64,
        help="Number of time samples for the sector-reduced backward pass.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-workers", dest="n_workers", type=int, default=_default_n_workers())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint-every", dest="checkpoint_every", type=int, default=100,
        help="Save a partial .npz every N trajectories.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
