"""Production run configuration: the explicit, frozen parameter surface.

Every scientific and algorithmic knob that affects a production result is a
field on :class:`ProductionConfig`.  Nothing is read from the environment at
run time by this entry point — the historical ``PPS_*`` environment variables
are deliberately NOT consulted, because the recorded production failure mode is
exactly a submit script whose environment silently disagreed with the driver
(TASK-2026-08-11-ALGRD, "Configuration discrepancies found").

Convention (Cut B): ``alpha + w = 1`` and ``lambda = alpha / (alpha + w)``,
so ``alpha = lambda`` and ``w = 1 - lambda``.  This matches
``pps_qj.parallel.grid_pps._alpha_w_from_lam`` exactly.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# Bump ALGORITHM_VERSION when the sampler's statistical output changes for a
# fixed seed.  Bump OUTPUT_SCHEMA_VERSION when the on-disk field set changes.
ALGORITHM_VERSION = "qjpps-prod-1.0.0"
OUTPUT_SCHEMA_VERSION = "2.0"

# Certified values.  See docs/PRODUCTION_ALGORITHM.md for the evidence behind
# each.  Changing one of these is a scientific decision, not a tuning knob.
_CERTIFIED_JUMP_UPDATE = "lowrank"
_CERTIFIED_SOLVER = "brentq"
_CERTIFIED_ENTROPY_STRIDE = 4
_CERTIFIED_PROPOSAL = "guided_reduced_rate"
_CERTIFIED_COMPENSATOR = "exact_radon_nikodym"
_CERTIFIED_RESAMPLING = "systematic_fixed_population"

_ALLOWED_JUMP_UPDATE = ("lowrank", "eigh")
_ALLOWED_SOLVER = ("brentq", "newton")
_ALLOWED_OBSERVABLES = (
    "CMI", "B_L", "entropy", "renyi", "activity", "corr_decay",
)
# Observables that are always computed; they cost nothing extra and the
# campaign charter requires the four CMI subsystem entropies to be stored
# separately rather than only their four-term difference.
_DEFAULT_OBSERVABLES = ("CMI", "B_L", "entropy", "activity")


class ConfigError(ValueError):
    """Raised when a production config is internally inconsistent."""


@dataclass
class ProductionConfig:
    """A fully-specified production cell.

    One config == one (L, zeta, lambda) cell run for ``realizations``
    independent realisations.
    """

    # --- physics ---------------------------------------------------------
    L: int
    zeta: float
    lam: float                      # lambda; alpha = lam, w = 1 - lam
    T: float
    N_c: int
    realizations: int = 5
    seed: int = 0

    # --- burn-in ---------------------------------------------------------
    # Fraction of the window sequence discarded before time-averaged
    # diagnostics (S_mean, chi_k, ...).  The t=T locators (CMI, B_L) are read
    # from the final population and are unaffected by this.
    n_burnin_frac: float = 0.25

    # --- sampler / discretisation ---------------------------------------
    # delta_tau = dtau_mult / (2 * alpha * (L - 1)).  dtau_mult = 6 is the
    # recorded production value of submit_clone_guided_prod.sh.
    dtau_mult: float = 6.0
    proposal_scheme: str = _CERTIFIED_PROPOSAL
    compensator: str = _CERTIFIED_COMPENSATOR
    resampling: str = _CERTIFIED_RESAMPLING

    # --- certified optimisations ----------------------------------------
    jump_update_method: str = _CERTIFIED_JUMP_UPDATE
    refresh_every: int = 100
    entropy_stride: int = _CERTIFIED_ENTROPY_STRIDE
    solver_method: str = _CERTIFIED_SOLVER
    eps_hazard: float = 1e-9

    # --- observables / output -------------------------------------------
    observables: tuple = field(default_factory=lambda: tuple(_DEFAULT_OBSERVABLES))
    record_selection_history: bool = False
    """Store the per-window selection index maps (N_c int32 per resampling event).

    OFF by default because the cost is real: ~1.7 MB per realisation at
    L = 128, T = 128, N_c = 128. ON, it is the only object from which the
    pairwise most-recent-common-ancestor distribution - and hence any
    genealogical variance estimate - can be reconstructed after the fact. The
    existing 20,355-run corpus cannot be re-diagnosed for any such question
    precisely because nothing like this was kept. It records no randomness and
    changes no result: paired-seed bitwise equality is asserted by
    tests/test_statistical_diagnostics.py."""
    output_dir: str = "outputs/production"
    run_label: str = ""
    n_workers: int = 1

    # --- free-form ------------------------------------------------------
    notes: str = ""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """alpha = lambda (Cut B convention alpha + w = 1)."""
        return float(self.lam)

    @property
    def w(self) -> float:
        """w = 1 - lambda (Cut B convention alpha + w = 1)."""
        return float(1.0 - self.lam)

    @property
    def delta_tau(self) -> float:
        base = 1.0 / max(2.0 * self.alpha * (self.L - 1), 1e-6)
        return float(self.dtau_mult) * base

    @property
    def n_steps(self) -> int:
        import math
        return max(1, int(math.ceil(self.T / self.delta_tau)))

    @property
    def n_burnin_steps(self) -> int:
        return int(self.n_steps * self.n_burnin_frac)

    @property
    def proposal_c(self) -> Optional[float]:
        """Guided proposal intensity factor c.  c = zeta for the certified
        reduced-rate proposal; None means the untilted physical proposal."""
        if self.proposal_scheme == _CERTIFIED_PROPOSAL:
            return float(self.zeta)
        if self.proposal_scheme == "physical":
            return None
        raise ConfigError(f"unknown proposal_scheme {self.proposal_scheme!r}")

    @property
    def record_renyi(self) -> bool:
        return "renyi" in self.observables or "corr_decay" in self.observables

    @property
    def computes_B_L(self) -> bool:
        """B_L / CMI need the Majorana tripartition, which needs L % 4 == 0."""
        return self.L % 4 == 0

    def realisation_seed(self, r: int) -> int:
        """Per-realisation seed.  Matches worker_clone_pps's stride so that a
        production cell can be compared against the historical corpus."""
        return int(self.seed) + int(r) * 999_983

    def run_id(self) -> str:
        label = self.run_label.strip()
        prefix = f"{label}_" if label else ""
        return (
            f"{prefix}L{self.L}_z{self.zeta:.4g}_lam{self.lam:.4g}"
            f"_T{self.T:.4g}_Nc{self.N_c}_s{self.seed}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.L < 4:
            raise ConfigError("L must be >= 4")
        if not (0.0 < self.zeta <= 1.0):
            raise ConfigError("zeta must be in (0, 1]")
        if not (0.0 < self.lam < 1.0):
            raise ConfigError(
                "lam must be in (0, 1); alpha = lam and w = 1 - lam must both "
                "be positive under the Cut B convention alpha + w = 1"
            )
        if self.T <= 0.0:
            raise ConfigError("T must be positive")
        if self.N_c < 1:
            raise ConfigError("N_c must be >= 1")
        if self.realizations < 1:
            raise ConfigError("realizations must be >= 1")
        if not (0.0 <= self.n_burnin_frac < 1.0):
            raise ConfigError("n_burnin_frac must be in [0, 1)")
        if self.dtau_mult <= 0.0:
            raise ConfigError("dtau_mult must be positive")
        if self.jump_update_method not in _ALLOWED_JUMP_UPDATE:
            raise ConfigError(
                f"jump_update_method must be one of {_ALLOWED_JUMP_UPDATE}"
            )
        if self.solver_method not in _ALLOWED_SOLVER:
            raise ConfigError(f"solver_method must be one of {_ALLOWED_SOLVER}")
        if self.entropy_stride < 1:
            raise ConfigError("entropy_stride must be >= 1")
        if self.refresh_every < 1:
            raise ConfigError("refresh_every must be >= 1")
        if self.compensator != _CERTIFIED_COMPENSATOR:
            raise ConfigError(
                "the only implemented compensator is "
                f"{_CERTIFIED_COMPENSATOR!r}; the guided path always applies "
                "the exact Radon-Nikodym weight exp[-(1-zeta) * dLambda]"
            )
        if self.resampling != _CERTIFIED_RESAMPLING:
            raise ConfigError(
                f"the only implemented resampling is {_CERTIFIED_RESAMPLING!r}"
            )
        bad = [o for o in self.observables if o not in _ALLOWED_OBSERVABLES]
        if bad:
            raise ConfigError(f"unknown observables: {bad}")
        if ("CMI" in self.observables or "B_L" in self.observables) \
                and not self.computes_B_L:
            raise ConfigError(
                f"CMI/B_L require L % 4 == 0 for the Majorana tripartition; "
                f"got L={self.L}"
            )

    def deviations_from_certified(self) -> list[str]:
        """Human-readable list of departures from the certified baseline.

        These are NOT errors — the entry point supports them — but every one
        is recorded in the output provenance so that a result run off-baseline
        can never later be mistaken for a baseline result.
        """
        out: list[str] = []
        if self.jump_update_method != _CERTIFIED_JUMP_UPDATE:
            out.append(
                f"jump_update_method={self.jump_update_method!r} "
                f"(certified: {_CERTIFIED_JUMP_UPDATE!r})"
            )
        if self.solver_method != _CERTIFIED_SOLVER:
            out.append(
                f"solver_method={self.solver_method!r} (certified: "
                f"{_CERTIFIED_SOLVER!r}; 'newton' is a STATISTICAL change and "
                f"has no production-scale paired-seed validation artifact)"
            )
        if self.entropy_stride != _CERTIFIED_ENTROPY_STRIDE:
            out.append(
                f"entropy_stride={self.entropy_stride} "
                f"(certified: {_CERTIFIED_ENTROPY_STRIDE})"
            )
        if self.proposal_scheme != _CERTIFIED_PROPOSAL:
            out.append(f"proposal_scheme={self.proposal_scheme!r}")
        if self.dtau_mult != 6.0:
            out.append(f"dtau_mult={self.dtau_mult} (recorded production: 6.0)")
        return out

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["observables"] = list(self.observables)
        return d

    def resolved_dict(self) -> dict[str, Any]:
        """Config plus every derived quantity, for the provenance record."""
        d = self.to_dict()
        d.update(
            alpha=self.alpha,
            w=self.w,
            delta_tau=self.delta_tau,
            n_steps=self.n_steps,
            n_burnin_steps=self.n_burnin_steps,
            proposal_c=self.proposal_c,
            record_renyi=self.record_renyi,
            run_id=self.run_id(),
            algorithm_version=ALGORITHM_VERSION,
            output_schema_version=OUTPUT_SCHEMA_VERSION,
            realisation_seeds=[
                self.realisation_seed(r) for r in range(self.realizations)
            ],
            deviations_from_certified=self.deviations_from_certified(),
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "observables" in kwargs:
            kwargs["observables"] = tuple(kwargs["observables"])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "ProductionConfig":
        """Load from YAML (if PyYAML is present) or JSON.

        PyYAML is optional on purpose: the production entry point must run on a
        cluster venv that carries only numpy/scipy.
        """
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ConfigError(
                    f"{p} is YAML but PyYAML is not installed. Either install "
                    f"PyYAML or supply the same config as JSON."
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ConfigError(f"{p} did not parse to a mapping")
        # Allow (and ignore) a leading comment-ish metadata block.
        data.pop("_comment", None)
        return cls.from_dict(data)
