"""Provenance capture for production runs.

The recorded failure this module exists to prevent: across the 15,139-run
historical corpus, ``git_commit``, ``seed``, ``burn_in`` and ``job_id`` are
absent from EVERY file (TASK-2026-08-14-C2CONV, NEXT_NUMERICS_QUESTION.md §5),
so no stored number can be tied to the code that produced it.

Nothing here reads secrets.  Environment capture is a strict allow-list of
scheduler variables; no general environment dump is taken.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from pps_qj.production.config import (
    ALGORITHM_VERSION,
    OUTPUT_SCHEMA_VERSION,
    ProductionConfig,
)

PROVENANCE_SCHEMA_VERSION = "1.0"

# Strict allow-list.  Scheduler identity only — never a general env dump, and
# never anything that could carry a credential.
_SLURM_ALLOWLIST = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NUM_NODES",
    "SLURM_CPUS_PER_TASK",
    "SLURM_NTASKS",
    "SLURM_SUBMIT_HOST",
    "SLURM_CLUSTER_NAME",
)

# Thread-pinning variables that materially affect timing, so they belong in the
# record next to the runtime numbers.
_THREAD_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Observable definitions, by canonical ID, so that a stored number can never
# later be read under the wrong convention.  OBS-BL-001 is RETIRED (one label
# covered two quantities) and is deliberately absent.
OBSERVABLE_DEFINITIONS: dict[str, dict[str, str]] = {
    "CMI": {
        "obs_id": "OBS-CMI-001",
        "definition": "S_AB + S_BC - S_B - S_ABC",
        "partition": (
            "quarter-system tripartition on Majorana indices [0, 3L/2): "
            "A=[0,L/2), B=[L/2,L), C=[L,3L/2)"
        ),
        "log_base": "2",
        "aggregation": "per-clone at t=T, then mean over clones, then mean over realisations",
    },
    "B_L": {
        "obs_id": "OBS-BLPROD-001",
        "definition": "B_L = CMI * S_AB, formed PER CLONE then averaged",
        "convention": (
            "average-of-products (ours). NOT OBS-BLKMR-001 "
            "(product-of-averages, KMR's). Never compare across them."
        ),
        "log_base": "2",
        "aggregation": "per-clone at t=T, then mean over clones, then mean over realisations",
    },
    "entropy": {
        "obs_id": "OBS-SHALF-TAVG-001",
        "definition": "half-cut von Neumann entropy S_{L/2}, tilted-weighted per window",
        "log_base": "2",
        "aggregation": (
            "weighted mean over clones per recorded window, then time-averaged "
            "over post-burn-in recorded windows. Recorded every entropy_stride "
            "windows; see entropy_stride."
        ),
    },
    "activity": {
        "obs_id": "OBS-ACTIVITY-001",
        "definition": "k_bar = <N_T>/(L*T); chi_k = Var(N_T^window)/(L*delta_tau)",
        "status_warning": "OBS-ACTIVITY-001 is needs_audit in research/state/",
        "aggregation": "post-burn-in time average of tilted-weighted window means",
    },
    "renyi": {
        "obs_id": "OBS-SHALF-FINAL-001",
        "definition": "half-cut Renyi entropies of index 2 and 3 at t=T",
        "log_base": "2",
        "aggregation": "mean over clones at t=T, then mean over realisations",
    },
    "corr_decay": {
        "obs_id": "unregistered",
        "definition": "translation-averaged C(r) = mean |<c_0^dag c_r>| over the final population",
        "status_warning": "no canonical OBS-* id; treat as diagnostic only",
    },
}


def _run_git(args: list[str], repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _repo_root() -> Path:
    # pps_qj/production/provenance.py -> repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def git_info() -> dict[str, Any]:
    """Commit, dirty flag, branch and short status.

    ``git_dirty`` is True when the working tree differs from HEAD.  A dirty
    production run is not blocked, but it IS flagged, and the changed paths are
    recorded so the result can be interpreted later.
    """
    repo = _repo_root()
    commit = _run_git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_branch": None,
            "git_describe": None,
            "git_dirty_paths": [],
            "git_available": False,
            "git_repo_root": str(repo),
        }
    status = _run_git(["status", "--porcelain"], repo) or ""
    dirty_paths = [ln.strip() for ln in status.splitlines() if ln.strip()]
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_paths),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "git_describe": _run_git(["describe", "--tags", "--always", "--dirty"], repo),
        # Cap the list: a pathological tree should not bloat every output file.
        "git_dirty_paths": dirty_paths[:200],
        "git_dirty_path_count": len(dirty_paths),
        "git_available": True,
        "git_repo_root": str(repo),
    }


def environment_info() -> dict[str, Any]:
    """Host, interpreter, library versions, scheduler identity, thread pinning."""
    import numpy as np

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
    }
    try:
        import scipy  # type: ignore
        info["scipy_version"] = scipy.__version__
    except ImportError:
        info["scipy_version"] = None

    # BLAS identity matters for reproducibility of timings and, at the last
    # bits, of results.
    try:
        cfg = np.__config__.show(mode="dicts")  # numpy >= 2
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        info["blas_name"] = blas.get("name")
        info["blas_version"] = blas.get("version")
    except Exception:
        info["blas_name"] = None
        info["blas_version"] = None

    scheduler = {k: os.environ[k] for k in _SLURM_ALLOWLIST if k in os.environ}
    info["scheduler"] = scheduler
    info["scheduler_job_id"] = (
        scheduler.get("SLURM_ARRAY_JOB_ID") or scheduler.get("SLURM_JOB_ID")
    )
    info["scheduler_task_id"] = scheduler.get("SLURM_ARRAY_TASK_ID")
    info["thread_env"] = {
        k: os.environ[k] for k in _THREAD_ALLOWLIST if k in os.environ
    }
    return info


def observable_definitions(observables) -> dict[str, dict[str, str]]:
    """The definition block for exactly the observables this run computed."""
    return {
        name: dict(OBSERVABLE_DEFINITIONS[name])
        for name in observables
        if name in OBSERVABLE_DEFINITIONS
    }


def build_provenance(
    cfg: ProductionConfig,
    *,
    started_at: float,
    wall_time: float,
    cpu_time: Optional[float],
    genealogy: dict[str, Any],
    per_realisation: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    """Assemble the full provenance record written beside every result."""
    resolved = cfg.resolved_dict()
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "code_version": ALGORITHM_VERSION,
        "entry_point": "pps_qj.production.run",
        "status": status,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)
        ),
        "timestamp_unix": float(started_at),
        "runtime_seconds": float(wall_time),
        "cpu_time_seconds": (None if cpu_time is None else float(cpu_time)),
        "git": git_info(),
        "environment": environment_info(),
        "config": resolved,
        "algorithm": {
            "family": "guided Feynman-Kac cloning / sequential Monte Carlo",
            "target_measure": (
                "QJ partial-post-selection tilted path measure; click weight "
                "zeta^n with partial post-selection parameter zeta"
            ),
            "proposal_scheme": cfg.proposal_scheme,
            "proposal_c": resolved["proposal_c"],
            "compensator": cfg.compensator,
            "compensator_formula": "exp[-(1 - zeta) * dLambda] per window",
            "waiting_time_solver": cfg.solver_method,
            "jump_update": cfg.jump_update_method,
            "low_rank_enabled": cfg.jump_update_method == "lowrank",
            "refresh_every": cfg.refresh_every,
            "entropy_stride": cfg.entropy_stride,
            "resampling": {
                "scheme": cfg.resampling,
                "population": cfg.N_c,
                "fixed_population": True,
                "trigger": "every window (unconditional)" if cfg.zeta != 1.0
                          else "disabled (zeta == 1)",
                "weights": "normalised before resampling",
            },
            "window": {
                "delta_tau": resolved["delta_tau"],
                "dtau_mult": cfg.dtau_mult,
                "n_steps": resolved["n_steps"],
            },
            "burn_in": {
                "n_burnin_frac": cfg.n_burnin_frac,
                "n_burnin_steps": resolved["n_burnin_steps"],
                "applies_to": (
                    "time-averaged diagnostics only; the t=T locators "
                    "(CMI, B_L) are read from the final population"
                ),
            },
            "deviations_from_certified_baseline":
                resolved["deviations_from_certified"],
        },
        "observable_definitions": observable_definitions(cfg.observables),
        "genealogy": genealogy,
        "per_realisation": per_realisation,
    }
