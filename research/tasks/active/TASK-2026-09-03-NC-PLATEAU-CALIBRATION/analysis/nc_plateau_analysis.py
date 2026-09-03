#!/usr/bin/env python3
"""THE analysis for TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

    .venv/bin/python3 analysis/nc_plateau_analysis.py

This is the ONLY place the frozen criteria are evaluated. ../SUCCESS_CRITERIA.yaml
and ../ANALYSIS_SPEC.yaml are the authority for what they are; this file is the
authority for nothing and merely computes them.

It runs to completion with zero, partial or complete results and says which. It
is written and validated BEFORE any of the new data exists, which is the point:
the criteria cannot be chosen after the numbers arrive.

STANDING RULES IT ENFORCES ON ITSELF (../ANALYSIS_SPEC.yaml)

  * every uncertainty is across INDEPENDENT POPULATIONS. Within-clone spread is
    a diagnostic. VIF is a VARIANCE-EQUIVALENCE diagnostic and is never used as
    a bias diagnostic, and never appears in any adequacy verdict.
  * founder count and genealogical ESS are diagnostics. Neither is ever read as
    a literal number of independent samples.
  * NO SMOOTHING. No interpolation replaces a measured point. No monotonicity is
    imposed. No lambda point is removed for being inconvenient. No value-based
    exclusion of any kind. The results file carries an audit block asserting all
    of this about the run that produced it.
  * finite-N_c movement is called DRIFT, never bias, because the N_c -> infinity
    target is not known.
  * a plateau is never inferred by eye, a 1/N_c fit is never forced, and no
    exponent is quoted from a ladder that has not passed its own asymptotic-form
    test.
  * increasing R never repairs finite-N_c drift and increasing N_c never repairs
    finite-R uncertainty. Section 1 reports the two budgets separately and every
    verdict downstream names which one binds.

Contains no scheduler call and cannot submit.
"""
import os, sys, json, glob, math, itertools, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
sys.path.insert(0, os.path.join(TASK, "tools"))
import design as D                                                  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
Z95 = 1.959963985
RNG = np.random.default_rng(20260903)
NBOOT = 4000

try:
    from scipy import stats as _st

    def chi2_sf(x, k):
        return float(_st.chi2.sf(x, k))
except Exception:                                    # pure-python fallback
    def chi2_sf(x, k):
        """Upper tail of chi2_k. Regularised incomplete gamma, series/CF."""
        a, xx = k / 2.0, x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1:
            s, t, n = 1.0 / a, 1.0 / a, 0
            while abs(t) > 1e-14 * abs(s) and n < 10000:
                n += 1
                t *= xx / (a + n)
                s += t
            return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b, c = xx + 1.0 - a, 1e300
        d = 1.0 / b
        h = d
        for i in range(1, 10000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-300:
                d = 1e-300
            c = b + an / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 1e-14:
                break
        return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


# ---------------------------------------------------------------------------
# Loading. Fresh populations from this task's arms; exact-compatible reused ones
# from the predecessor trees named in tools/build_arms.REUSE. Nothing is read
# from any predecessor SUMMARY: only from the JSON the sampler itself wrote.
# ---------------------------------------------------------------------------
def load():
    pops, srcs = [], collections.Counter()
    pats = [os.path.join(TASK, "*", "results", "*.json"),
            os.path.join(TASK, "conditional", "*", "results", "*.json"),
            os.path.join(REPO, "research", "tasks", "**", "results", "*.json")]
    seen = set()
    for pat in pats:
        for p in sorted(glob.glob(pat, recursive=True)):
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if not isinstance(d, dict) or "cmi_weighted_mean" not in d:
                continue
            if d.get("status") not in (None, "ok"):
                continue
            parts = os.path.relpath(p, REPO).split(os.sep)
            task = next((x for x in parts if x.startswith("TASK-")), "?")
            pops.append(dict(
                task=task, L=int(d["L"]), T=float(d["T"]), zeta=float(d["zeta"]),
                lam=round(float(d["lam"]), 6), N_c=int(d["N_c"]),
                dm=float(d["dtau_mult"]), seed=int(d["seed"]),
                scheme=d.get("resample_scheme", "systematic"),
                mean=float(d["cmi_weighted_mean"]),
                within=float(d.get("cmi_within_var", np.nan)),
                n_steps=int(d["n_steps"]),
                wall=float(d.get("wall_s", np.nan)),
                anc=int(d.get("n_distinct_anc_final", 0)),
                lwv=d.get("logw_carry_var_final")))
            srcs[task] += 1
    return pops, srcs


def cellkey(p):
    return (p["L"], p["T"], p["zeta"], p["lam"], p["N_c"], p["dm"])


class Cell:
    """One (L, T, zeta, lambda, N_c, dtau_mult) cell. dtau_mult is part of the
    identity: campaign E varies it, and pooling two discretisations into one
    cell is the single easiest way to manufacture a wrong answer here."""

    def __init__(self, k, pops):
        self.k = k
        self.pops = sorted(pops, key=lambda p: p["seed"])   # observable-blind
        self.m = np.array([p["mean"] for p in self.pops])
        self.w = np.array([p["within"] for p in self.pops])

    L = property(lambda s: s.k[0])
    lam = property(lambda s: s.k[3])
    N_c = property(lambda s: s.k[4])
    dm = property(lambda s: s.k[5])
    R = property(lambda s: s.m.size)
    mean = property(lambda s: float(s.m.mean()))
    sd = property(lambda s: float(s.m.std(ddof=1)) if s.m.size > 1 else float("nan"))
    sem = property(lambda s: s.sd / math.sqrt(s.m.size) if s.m.size > 1
                   else float("nan"))
    tasks = property(lambda s: sorted({p["task"] for p in s.pops}))

    def block(self, i, nb=2):
        """Disjoint block i of nb, cut in SEED ORDER, which is observable-blind.
        Cutting on anything the observable can see would let the block test pass
        by construction."""
        n = self.R // nb
        return self.m[i * n:(i + 1) * n]

    @property
    def vif(self):
        """VARIANCE EQUIVALENCE ONLY. Never a bias diagnostic."""
        wm = float(np.nanmean(self.w))
        v = float(np.var(self.m, ddof=1)) if self.R > 1 else float("nan")
        return v * self.N_c / wm if wm > 0 else float("nan")


def build(pops):
    g = collections.defaultdict(list)
    for p in pops:
        g[cellkey(p)].append(p)
    return {k: Cell(k, v) for k, v in g.items()}


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def delta(cell_lo, cell_hi):
    """Delta_N = I_2N - I_N with its across-population standard error.

    The two rungs are independent populations, so the errors add in quadrature.
    Delta is DRIFT, not bias: the N_c -> infinity target is unknown, so a
    non-zero Delta says the estimate is still moving and nothing about which
    direction the truth lies in.
    """
    d = cell_hi.mean - cell_lo.mean
    s = math.hypot(cell_lo.sem, cell_hi.sem)
    return d, s


def ci(d, s):
    return (d - Z95 * s, d + Z95 * s)


def inside(lo, hi, tau):
    return (-tau <= lo) and (hi <= tau)


def wls_1overN(Ns, ys, sems):
    """I_N = I_inf + B/N by weighted least squares. TWO parameters, so a
    3-rung ladder has 1 dof and a 4-rung ladder 2. Returns the fit and its
    chi2 p-value; the p-value is a REJECTION test of the 1/N form, never
    evidence for it."""
    x = 1.0 / np.asarray(Ns, float)
    y = np.asarray(ys, float)
    w = 1.0 / np.asarray(sems, float) ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    den = S * Sxx - Sx * Sx
    if abs(den) < 1e-300:
        return None
    B = (S * Sxy - Sx * Sy) / den
    I = (Sxx * Sy - Sx * Sxy) / den
    r = y - (I + B * x)
    chi2 = float((w * r * r).sum())
    dof = len(x) - 2
    return dict(I_inf=I, B=B, chi2=chi2, dof=dof,
                p=chi2_sf(chi2, dof) if dof > 0 else None,
                se_I=math.sqrt(Sxx / den), se_B=math.sqrt(S / den),
                resid=[float(v) for v in r])


def fit_free_gamma(Ns, ys, sems, grid=None):
    """I_N = I_inf + B * N**-gamma, gamma scanned. THREE parameters, so it needs
    at least four rungs to have any dof at all -- a three-rung ladder fits it
    exactly and the fit means nothing. Reported with its dof so that cannot be
    misread."""
    if grid is None:
        grid = np.linspace(0.05, 3.0, 1181)
    Ns = np.asarray(Ns, float)
    y = np.asarray(ys, float)
    w = 1.0 / np.asarray(sems, float) ** 2
    best = None
    for g in grid:
        x = Ns ** (-g)
        S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
        Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
        den = S * Sxx - Sx * Sx
        if abs(den) < 1e-300:
            continue
        B = (S * Sxy - Sx * Sy) / den
        I = (Sxx * Sy - Sx * Sxy) / den
        r = y - (I + B * x)
        c2 = float((w * r * r).sum())
        if best is None or c2 < best["chi2"]:
            best = dict(gamma=float(g), I_inf=I, B=B, chi2=c2,
                        dof=len(Ns) - 3,
                        p=chi2_sf(c2, len(Ns) - 3) if len(Ns) > 3 else None)
    return best


def b_eff(cell_lo, cell_hi):
    """B_eff(N) = -2 N (I_2N - I_N).

    If I_N = I_inf + B/N exactly then B_eff(N) == B for every N. A B_eff that
    still moves across the top of a ladder is evidence the ladder is
    PRE-ASYMPTOTIC, and no coefficient may be quoted from it.
    """
    d, s = delta(cell_lo, cell_hi)
    N = cell_lo.N_c
    return -2.0 * N * d, 2.0 * N * s


def r_required(sd_lo, sd_hi, tau, z=Z95):
    """Matched R needed for a Delta half-width of tau. This is the R BUDGET and
    it is completely separate from the N_c budget: no R makes finite-N_c drift
    smaller, and no N_c makes a small-R interval narrower."""
    if not (np.isfinite(sd_lo) and np.isfinite(sd_hi)):
        return None
    return int(math.ceil((z / tau) ** 2 * (sd_lo ** 2 + sd_hi ** 2)))


# ---------------------------------------------------------------------------
# Curve and crossing quality (../ANALYSIS_SPEC.yaml section "curve_quality")
# ---------------------------------------------------------------------------
def curve(cells, L, N_c, lams, dm=6.0, zeta=0.35):
    out = []
    for l in lams:
        c = cells.get((L, float(L), zeta, round(l, 6), N_c, dm))
        out.append(c)
    return out


def curve_quality(lams, cs):
    have = [(l, c) for l, c in zip(lams, cs) if c is not None and c.R > 1]
    if len(have) < 3:
        return dict(status="insufficient", n_points=len(have))
    x = np.array([l for l, _ in have])
    y = np.array([c.mean for _, c in have])
    e = np.array([c.sem for _, c in have])
    d1 = np.diff(y)
    d1e = np.hypot(e[:-1], e[1:])
    d2 = np.diff(y, 2)
    res = []
    for i, (l, c) in enumerate(have):
        a, b = c.block(0), c.block(1)
        sh = (float(a.mean() - b.mean()),
              math.hypot(a.std(ddof=1) / math.sqrt(a.size),
                         b.std(ddof=1) / math.sqrt(b.size))) \
            if min(a.size, b.size) >= 2 else (float("nan"), float("nan"))
        loo = [float(np.delete(c.m, j).mean()) for j in range(c.R)]
        z = (c.m - c.m.mean()) / (c.sd if c.sd > 0 else 1.0)
        res.append(dict(lam=float(l), R=c.R, mean=c.mean, sem=c.sem,
                        sd=c.sd, vif_variance_diagnostic_only=c.vif,
                        split_half_diff=sh[0], split_half_sem=sh[1],
                        loo_min=min(loo), loo_max=max(loo),
                        loo_spread_in_sem=(max(loo) - min(loo)) / c.sem
                        if c.sem > 0 else float("nan"),
                        max_abs_z=float(np.abs(z).max()),
                        n_over_3sigma=int((np.abs(z) > 3).sum())))
    return dict(status="ok", n_points=len(have),
                lams=[float(v) for v in x], means=[float(v) for v in y],
                sems=[float(v) for v in e],
                increments=[float(v) for v in d1],
                increment_sems=[float(v) for v in d1e],
                increments_resolved=[bool(abs(a) > Z95 * b)
                                     for a, b in zip(d1, d1e)],
                second_differences=[float(v) for v in d2],
                roughness=float(np.sqrt(np.mean(d2 ** 2))) if d2.size else None,
                roughness_in_sem=float(np.sqrt(np.mean(d2 ** 2)) / np.mean(e))
                if d2.size else None,
                per_point=res)


def crossings(lams, ya, ea, yb, eb, nboot=NBOOT, popsa=None, popsb=None):
    """Cross-L difference D = I_a - I_b on a shared raw grid.

    Reports raw sign changes, resolved sign changes, a bootstrap crossing-count
    histogram and interval, and whether the crossing is ENDPOINT-INDUCED -- the
    failure the predecessor task was built to fix. NOTHING IS SMOOTHED and no
    point is dropped; interpolation is used ONLY to place a crossing between two
    measured points that bracket it, never to replace a measurement.
    """
    x = np.asarray(lams, float)
    Da = np.asarray(ya) - np.asarray(yb)
    Se = np.hypot(np.asarray(ea), np.asarray(eb))
    sign = np.sign(Da)
    raw = [i for i in range(len(x) - 1) if sign[i] * sign[i + 1] < 0]
    resolved = [i for i in raw
                if abs(Da[i]) > Z95 * Se[i] and abs(Da[i + 1]) > Z95 * Se[i + 1]]

    def loc(i, d):
        return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))

    locs = [loc(i, Da) for i in raw]
    boot = []
    for _ in range(nboot):
        db = Da + RNG.normal(0.0, Se)
        s = np.sign(db)
        ii = [i for i in range(len(x) - 1) if s[i] * s[i + 1] < 0]
        boot.append((len(ii), [loc(i, db) for i in ii]))
    counts = collections.Counter(n for n, _ in boot)
    allloc = [v for _, ls in boot for v in ls]
    endpoint = bool(locs) and any(
        (l - x[0]) < (x[1] - x[0]) or (x[-1] - l) < (x[-1] - x[-2]) for l in locs)
    return dict(
        D=[float(v) for v in Da], D_sem=[float(v) for v in Se],
        raw_sign_changes=len(raw), resolved_sign_changes=len(resolved),
        crossing_lambdas=locs,
        bootstrap_count_histogram={str(k): int(v) for k, v in sorted(counts.items())},
        fraction_exactly_one=float(counts.get(1, 0)) / nboot,
        bootstrap_ci=[float(np.percentile(allloc, 2.5)),
                      float(np.percentile(allloc, 97.5))] if allloc else None,
        bootstrap_median=float(np.median(allloc)) if allloc else None,
        endpoint_induced=endpoint,
        grid=[float(v) for v in x])


def loo_lambda_crossing(lams, ya, ea, yb, eb):
    """Leave-one-lambda-out stability of the crossing. A crossing that survives
    only on the full grid is a crossing that one measured point is carrying."""
    out = []
    for j in range(len(lams)):
        keep = [i for i in range(len(lams)) if i != j]
        if len(keep) < 3:
            continue
        r = crossings([lams[i] for i in keep], [ya[i] for i in keep],
                      [ea[i] for i in keep], [yb[i] for i in keep],
                      [eb[i] for i in keep], nboot=400)
        out.append(dict(dropped_lambda=float(lams[j]),
                        raw_sign_changes=r["raw_sign_changes"],
                        crossings=r["crossing_lambdas"],
                        endpoint_induced=r["endpoint_induced"]))
    return out


# ---------------------------------------------------------------------------
# Plateau criteria P1-P5 (../SUCCESS_CRITERIA.yaml). Frozen before any datum.
# ---------------------------------------------------------------------------
def plateau(ladder, tau_I=D.TAU_I):
    """ladder: {N_c: Cell}, at one (L, lambda, dtau_mult)."""
    Ns = sorted(ladder)
    steps = []
    for N in Ns:
        if 2 * N in ladder:
            lo, hi = ladder[N], ladder[2 * N]
            d, s = delta(lo, hi)
            l95, u95 = ci(d, s)
            be, bse = b_eff(lo, hi)
            steps.append(dict(
                N=N, N2=2 * N, R_lo=lo.R, R_hi=hi.R, I_lo=lo.mean, I_hi=hi.mean,
                delta=d, sem=s, ci=[l95, u95],
                P1_compatible_with_zero=bool(abs(d) <= Z95 * s),
                P2_ci_inside_tau_I=bool(inside(l95, u95, tau_I)),
                half_width=Z95 * s, half_width_over_tau=Z95 * s / tau_I,
                B_eff=be, B_eff_sem=bse,
                R_required_for_P2=r_required(lo.sd, hi.sd, tau_I)))
    v = dict(rungs=Ns, steps=steps, tau_I=tau_I)
    if not steps:
        v.update(verdict="NO_STEP", reason="the ladder has no N -> 2N pair")
        return v

    top = steps[-1]
    # P3: successive |Delta| must not increase materially.
    if len(steps) >= 2:
        a, b = steps[-2], steps[-1]
        inc = abs(b["delta"]) - abs(a["delta"])
        v["P3_no_material_increase"] = bool(
            inc <= Z95 * math.hypot(a["sem"], b["sem"]))
        v["P3_increase"] = inc
    else:
        v["P3_no_material_increase"] = None
        v["P3_reason"] = "only one step; P3 needs two"

    # P4: independent-population block estimates agree at both top rungs.
    p4, p4d = True, []
    for N in (top["N"], top["N2"]):
        c = ladder[N]
        if c.R >= 4:
            a, b = c.block(0), c.block(1)
            dd = float(a.mean() - b.mean())
            ss = math.hypot(a.std(ddof=1) / math.sqrt(a.size),
                            b.std(ddof=1) / math.sqrt(b.size))
            ok = abs(dd) <= Z95 * ss
            p4 &= ok
            p4d.append(dict(N_c=N, diff=dd, sem=ss, agree=bool(ok)))
        else:
            p4 = False
            p4d.append(dict(N_c=N, diff=None, sem=None, agree=False,
                            reason=f"R = {c.R} < 4, blocks have no variance"))
    v["P4_blocks_agree"] = bool(p4)
    v["P4_detail"] = p4d

    # P5: the conclusion survives dropping the lowest included rung.
    if len(Ns) >= 3:
        sub = {n: ladder[n] for n in Ns[1:]}
        s5 = plateau(sub, tau_I)
        v["P5_survives_dropping_lowest"] = bool(
            s5.get("verdict") == "PLATEAU_OBSERVED")
        v["P5_verdict_without_lowest"] = s5.get("verdict")
    else:
        v["P5_survives_dropping_lowest"] = None
        v["P5_reason"] = "fewer than three rungs; dropping one leaves no step"

    passed = [top["P1_compatible_with_zero"], top["P2_ci_inside_tau_I"],
              v.get("P3_no_material_increase"), v.get("P4_blocks_agree"),
              v.get("P5_survives_dropping_lowest")]
    if all(p is True for p in passed):
        v["verdict"] = "PLATEAU_OBSERVED"
    elif top["P1_compatible_with_zero"] and not top["P2_ci_inside_tau_I"]:
        # The distinction the whole campaign turns on. A step that is
        # compatible with zero but whose interval is wider than the material
        # tolerance has NOT demonstrated convergence: it has demonstrated that
        # R was too small to tell. Increasing N_c does not fix this.
        v["verdict"] = "UNRESOLVED_R_LIMITED"
        v["binding_budget"] = "R"
        v["R_required_for_P2"] = top["R_required_for_P2"]
    elif not top["P1_compatible_with_zero"]:
        v["verdict"] = "STILL_DRIFTING"
        v["binding_budget"] = "N_c"
    else:
        v["verdict"] = "UNRESOLVED"
    v["criteria"] = dict(P1=passed[0], P2=passed[1], P3=passed[2],
                         P4=passed[3], P5=passed[4])
    return v


def asymptotic_form(ladder):
    """Section 3B. Test 1/N and free-gamma SEPARATELY, and refuse to quote a
    coefficient from a ladder that fails its own stability test."""
    Ns = sorted(ladder)
    if len(Ns) < 3:
        return dict(status="insufficient", rungs=Ns,
                    reason="a 1/N fit over two rungs has zero dof and is not a test")
    y = [ladder[n].mean for n in Ns]
    e = [ladder[n].sem for n in Ns]
    out = dict(rungs=Ns, means=y, sems=e,
               fit_1overN=wls_1overN(Ns, y, e),
               fit_free_gamma=fit_free_gamma(Ns, y, e))
    if len(Ns) >= 4:
        out["fit_1overN_drop_lowest"] = wls_1overN(Ns[1:], y[1:], e[1:])
    beffs = []
    for n in Ns:
        if 2 * n in ladder:
            b, s = b_eff(ladder[n], ladder[2 * n])
            beffs.append(dict(N=n, B_eff=b, sem=s))
    out["B_eff"] = beffs
    stable = None
    if len(beffs) >= 2:
        a, b = beffs[-2], beffs[-1]
        stable = bool(abs(a["B_eff"] - b["B_eff"])
                      <= Z95 * math.hypot(a["sem"], b["sem"]))
    out["B_eff_stable_on_top_two"] = stable
    f = out["fit_1overN"]
    rejected = bool(f and f["p"] is not None and f["p"] < 0.05)
    out["1overN_rejected_at_5pct"] = rejected
    if rejected or stable is not True:
        out["verdict"] = "NO_OBSERVED_1_OVER_NC_ASYMPTOTIC_REGIME"
        out["coefficient_B"] = None
        out["why"] = ("the 1/N form is rejected by its own chi2" if rejected
                      else "B_eff is not stable across the top two steps"
                      if stable is False else
                      "there are not two B_eff points to compare")
    else:
        out["verdict"] = "LOCAL_1_OVER_NC_REGIME_CONSISTENT"
        out["coefficient_B"] = dict(
            B=f["B"], se=f["se_B"],
            stable_over=[beffs[-2]["N"], beffs[-1]["N"] * 2],
            scope="LOCAL and CELL-SPECIFIC. Not a law B(L, zeta, lambda). "
                  "Not transferable to another observable, L, zeta or lambda.")
    return out


# ---------------------------------------------------------------------------
# Figures. No smoothing anywhere; markers are measurements, lines only connect
# adjacent measurements.
# ---------------------------------------------------------------------------
def figures(cells, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [figures skipped: {e}]")
        return []
    made = []

    def save(fig, name, title):
        fig.suptitle(title + "   —   raw measurements, no smoothing", fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIG, name)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(name)

    def ladder_of(L, lam, dm=6.0):
        return {k[4]: c for k, c in cells.items()
                if k[0] == L and abs(k[3] - lam) < 1e-9 and k[5] == dm}

    # A / B / C: the L = 64 central ladder
    lad = ladder_of(64, D.A_LAM)
    if len(lad) >= 2:
        Ns = sorted(lad)
        f, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(Ns, [lad[n].mean for n in Ns], yerr=[lad[n].sem for n in Ns],
                    marker="o", capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$N_c$")
        ax.set_ylabel("CMI")
        ax.grid(alpha=.3)
        save(f, "figureA_L64_central_vs_Nc.png",
             f"FIGURE A  L=64 T=64 zeta=0.35 lambda={D.A_LAM}  CMI vs $N_c$")

        st = [(n, *delta(lad[n], lad[2 * n])) for n in Ns if 2 * n in lad]
        if st:
            f, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar([s[0] for s in st], [s[1] for s in st],
                        yerr=[Z95 * s[2] for s in st], marker="o", capsize=3)
            ax.axhline(0, color="k", lw=.8)
            ax.axhspan(-D.TAU_I, D.TAU_I, color="green", alpha=.12,
                       label=f"$\\pm\\tau_I$ = {D.TAU_I}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("$N$")
            ax.set_ylabel("$\\Delta_N = I_{2N}-I_N$")
            ax.legend(fontsize=8)
            ax.grid(alpha=.3)
            save(f, "figureB_L64_deltaN.png",
                 "FIGURE B  L=64  $\\Delta_N$ with 95% intervals against "
                 "the frozen $\\tau_I$")

            f, ax = plt.subplots(figsize=(6, 4))
            be = [(n, *b_eff(lad[n], lad[2 * n])) for n in Ns if 2 * n in lad]
            ax.errorbar([b[0] for b in be], [b[1] for b in be],
                        yerr=[Z95 * b[2] for b in be], marker="s", capsize=3)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("$N$")
            ax.set_ylabel("$B_{eff}(N) = -2N\\Delta_N$")
            ax.grid(alpha=.3)
            save(f, "figureC_L64_Beff.png",
                 "FIGURE C  L=64  $B_{eff}$ — flat only if $I_N=I_\\infty+B/N$")

    # D / E / F: the transition-region grid
    have = {N: curve(cells, 64, N, D.B_GRID) for N in D.B_NCS}
    if any(any(c is not None for c in v) for v in have.values()):
        f, ax = plt.subplots(figsize=(6.5, 4.2))
        for N, cs in have.items():
            pts = [(l, c) for l, c in zip(D.B_GRID, cs) if c is not None]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1].mean for p in pts],
                            yerr=[p[1].sem for p in pts], marker="o", capsize=2,
                            label=f"$N_c$={N}")
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("CMI")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        save(f, "figureD_L64_region_curves.png",
             "FIGURE D  L=64 transition-region 7-point curves")

        f, ax = plt.subplots(figsize=(6.5, 4.2))
        for lo, hi in ((512, 1024), (1024, 2048)):
            pts = [(l, a, b) for l, a, b in zip(D.B_GRID, have.get(lo, []),
                                                have.get(hi, []))
                   if a is not None and b is not None]
            if pts:
                dd = [delta(p[1], p[2]) for p in pts]
                ax.errorbar([p[0] for p in pts], [d[0] for d in dd],
                            yerr=[Z95 * d[1] for d in dd], marker="o", capsize=2,
                            label=f"$\\Delta_{{{lo}\\to{hi}}}$")
        ax.axhline(0, color="k", lw=.8)
        ax.axhspan(-D.TAU_I, D.TAU_I, color="green", alpha=.12)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$\\Delta$")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        save(f, "figureE_L64_delta_vs_lambda.png",
             "FIGURE E  finite-$N_c$ increments across the region")

        hh = res.get("campaign_B", {}).get("shape", {})
        if hh:
            f, ax = plt.subplots(figsize=(6.5, 4.2))
            for step, v in hh.items():
                if v.get("status") != "ok":
                    continue
                ax.plot(v["lams"], v["delta"], "o-", label=f"{step} additive resid")
                ax.plot(v["lams"], v["ratio_centred"], "s--",
                        label=f"{step} multiplicative resid")
            ax.axhline(0, color="k", lw=.8)
            ax.set_xlabel("$\\lambda$")
            ax.set_ylabel("residual about the fitted constant")
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
            save(f, "figureF_additive_vs_multiplicative.png",
                 "FIGURE F  H1 additive vs H2 multiplicative")

    # G / H: the three central ladders together
    lads = {L: ladder_of(L, 0.3032) for L in (64, 96, 128)}
    if any(len(v) >= 2 for v in lads.values()):
        f, ax = plt.subplots(figsize=(6.5, 4.2))
        for L, lad2 in lads.items():
            Ns = sorted(lad2)
            if Ns:
                ax.errorbar(Ns, [lad2[n].mean for n in Ns],
                            yerr=[lad2[n].sem for n in Ns], marker="o",
                            capsize=2, label=f"L={L}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$N_c$")
        ax.set_ylabel("CMI")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        save(f, "figureG_central_ladders.png",
             "FIGURE G  central ladders at L = 64, 96, 128, lambda = 0.3032")

        f, ax = plt.subplots(figsize=(6.5, 4.2))
        for L, lad2 in lads.items():
            Ns = sorted(lad2)
            st = [(n, *delta(lad2[n], lad2[2 * n])) for n in Ns if 2 * n in lad2]
            if st:
                ax.plot([s[0] for s in st],
                        [Z95 * s[2] / D.TAU_I for s in st], "o-", label=f"L={L} HW/$\\tau_I$")
                ax.plot([s[0] for s in st],
                        [abs(s[1]) / D.TAU_I for s in st], "s--",
                        label=f"L={L} $|\\Delta|/\\tau_I$")
        ax.axhline(1.0, color="k", lw=.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("$N$")
        ax.set_ylabel("in units of the frozen $\\tau_I$")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        save(f, "figureH_adequacy_vs_L.png",
             "FIGURE H  adequacy: drift and resolution against $\\tau_I$, by L")

    # I / J: the discretisation experiment
    for N in D.E_NCS:
        pts = [(dm, cells.get((64, 64.0, 0.35, round(D.E_LAM, 6), N, dm)))
               for dm in D.E_DTAUS]
        pts = [(dm, c) for dm, c in pts if c is not None]
        if len(pts) >= 2:
            f, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar([p[0] for p in pts], [p[1].mean for p in pts],
                        yerr=[p[1].sem for p in pts], marker="o", capsize=3)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("dtau_mult  (K = 816 / 408 / 204)")
            ax.set_ylabel("CMI")
            ax.grid(alpha=.3)
            save(f, f"figure{'I' if N == 64 else 'J'}_dtau_nc{N}.png",
                 f"FIGURE {'I' if N == 64 else 'J'}  discretisation test, "
                 f"$N_c$={N} — target measure exactly unchanged")

    # K / L: the higher-L mock scans and cross-L differences, when they exist
    k_any = False
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    for L in (32, 48, 64, 96, 128):
        for N in (512, 1024, 2048, 4096):
            cs = curve(cells, L, N, D.MOCK9_GRID)
            pts = [(l, c) for l, c in zip(D.MOCK9_GRID, cs) if c is not None]
            if len(pts) >= 5:
                k_any = True
                ax.errorbar([p[0] for p in pts], [p[1].mean for p in pts],
                            yerr=[p[1].sem for p in pts], marker="o", capsize=2,
                            label=f"L={L} $N_c$={N}")
    if k_any:
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("CMI")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        save(f, "figureK_higherL_mock_curves.png",
             "FIGURE K  higher-L mock-production curves")
    else:
        plt.close(f)

    cr = res.get("locator", {}).get("pairs", {})
    if cr:
        f, ax = plt.subplots(figsize=(6.5, 4.2))
        for name, v in cr.items():
            if v.get("status") != "ok":
                continue
            ax.errorbar(v["grid"], v["D"], yerr=v["D_sem"], marker="o",
                        capsize=2, label=name)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$D = I_{L_1} - I_{L_2}$")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        save(f, "figureL_crossL_differences.png",
             "FIGURE L  cross-L differences near the locator region")
    return made


# ---------------------------------------------------------------------------
def main():
    pops, srcs = load()
    cells = build(pops)
    res = dict(task="TASK-2026-09-03-NC-PLATEAU-CALIBRATION",
               tau_I=D.TAU_I, tau_lambda=D.TAU_LAMBDA, tau_D=D.TAU_D,
               audit=dict(
                   smoothing_applied=False,
                   interpolation_replacing_a_measured_point=False,
                   monotonicity_imposed=False,
                   lambda_points_removed=0,
                   value_based_exclusions=0,
                   uncertainty_source="across independent populations",
                   vif_used_as_bias_diagnostic=False,
                   founder_count_used_as_sample_size=False,
                   finite_Nc_movement_called="drift, never bias"),
               populations=len(pops), cells=len(cells),
               sources={k: v for k, v in sorted(srcs.items())})

    print("=" * 78)
    print("TASK-2026-09-03-NC-PLATEAU-CALIBRATION — analysis")
    print("=" * 78)
    print(f"{len(pops)} populations in {len(cells)} cells")
    for k, v in sorted(srcs.items()):
        print(f"    {v:>5}  {k}")
    if not pops:
        print("\nNO RESULTS ON DISK. Everything below is empty by construction, "
              "not by failure.")

    def lad(L, lam, dm=6.0, zeta=0.35):
        return {k[4]: c for k, c in cells.items()
                if k[0] == L and abs(k[3] - lam) < 1e-9 and k[5] == dm
                and abs(k[2] - zeta) < 1e-12}

    # -- 1. N_c versus R, kept separate ------------------------------------
    print("\n" + "-" * 78)
    print("1. THE TWO BUDGETS. N_c controls finite-particle drift; R controls "
          "the\n   uncertainty of the finite-N_c population mean. Neither "
          "repairs the other.")
    print("-" * 78)
    print(f"{'L':>4}{'lam':>8}{'dm':>5}{'N_c':>7}{'R':>5}{'mean':>10}{'sd':>9}"
          f"{'SEM':>9}{'R for tau_I':>12}  source")
    budget = []
    for k in sorted(cells):
        c = cells[k]
        rr = r_required(c.sd, c.sd, D.TAU_I)
        print(f"{c.L:>4}{c.lam:>8.4f}{c.dm:>5g}{c.N_c:>7}{c.R:>5}{c.mean:>10.5f}"
              f"{c.sd:>9.5f}{c.sem:>9.5f}{(rr if rr else 0):>12}  "
              f"{'+'.join(t.replace('TASK-2026-', '') for t in c.tasks)}")
        budget.append(dict(L=c.L, lam=c.lam, dtau_mult=c.dm, N_c=c.N_c, R=c.R,
                           mean=c.mean, sd=c.sd, sem=c.sem,
                           R_for_matched_delta_at_tau_I=rr,
                           vif_variance_diagnostic_only=c.vif,
                           tasks=c.tasks))
    res["budgets"] = budget

    # -- 2. Campaign A ------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. CAMPAIGN A — deep central ladder, L = 64, lambda = 0.3032")
    print("-" * 78)
    A = lad(64, D.A_LAM)
    res["campaign_A"] = dict(plateau=plateau(A), asymptotic=asymptotic_form(A))
    report_ladder(A, res["campaign_A"])

    # -- 3. Campaign B ------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. CAMPAIGN B — transition-region shape, L = 64")
    print("-" * 78)
    B = {}
    for N in D.B_NCS:
        cs = curve(cells, 64, N, D.B_GRID)
        B[N] = cs
        q = curve_quality(D.B_GRID, cs)
        res.setdefault("campaign_B", {}).setdefault("quality", {})[str(N)] = q
        print(f"  N_c={N}: {q['n_points']} of 7 lambdas measured"
              + (f", roughness {q['roughness']:.5f} "
                 f"({q['roughness_in_sem']:.2f} SEM)" if q["status"] == "ok"
                 else ""))
        if q["status"] == "ok":
            for p in q["per_point"]:
                print(f"      lam={p['lam']:.4f}  R={p['R']:<3} "
                      f"CMI={p['mean']:.5f} +- {p['sem']:.5f}  "
                      f"split-half {p['split_half_diff']:+.5f} "
                      f"+- {p['split_half_sem']:.5f}  "
                      f"LOO spread {p['loo_spread_in_sem']:.2f} SEM  "
                      f"max|z| {p['max_abs_z']:.2f}")
    res["campaign_B"]["shape"] = shape_tests(B)
    report_shape(res["campaign_B"]["shape"])

    # -- 4. Locator convergence --------------------------------------------
    print("\n" + "-" * 78)
    print("4. LOCATOR CONVERGENCE — does the CROSSING converge before the "
          "absolute level?")
    print("-" * 78)
    res["locator"] = locator(cells)
    report_locator(res["locator"])

    # -- 5. Campaigns C and D ----------------------------------------------
    print("\n" + "-" * 78)
    print("5. CAMPAIGNS C and D — L = 96 and L = 128 central ladders")
    print("-" * 78)
    for L, tag in ((96, "campaign_C"), (128, "campaign_D")):
        lz = lad(L, 0.3032)
        res[tag] = dict(plateau=plateau(lz), asymptotic=asymptotic_form(lz))
        print(f"\n  L = {L}")
        report_ladder(lz, res[tag], indent="  ")

    # -- 6. Campaign E ------------------------------------------------------
    print("\n" + "-" * 78)
    print("6. CAMPAIGN E — discretisation / continuous-time particle limit")
    print("-" * 78)
    res["campaign_E"] = discretisation(cells)
    report_E(res["campaign_E"])

    # -- 7. Figures ---------------------------------------------------------
    res["figures"] = figures(cells, res)
    print(f"\n{len(res['figures'])} figure(s) written to analysis/figures/")

    dest = os.path.join(TASK, "NC_PLATEAU_RESULTS.json")
    json.dump(res, open(dest, "w"), indent=1, default=float)
    print(f"wrote {dest}")
    print("\nNothing here closes a dispute, promotes a claim, or writes "
          "research/state/**.")
    print("Write FALSIFICATION_RESULTS.md against the FROZEN "
          "FALSIFICATION_PLAN.md.")
    print("The plan is frozen; the results are a different file. Do not edit "
          "the plan to match what happened.")
    return 0


def report_ladder(ladder, r, indent=""):
    p = r["plateau"]
    if not ladder:
        print(indent + "  no populations at this cell yet.")
        return
    print(indent + f"  rungs {sorted(ladder)}")
    for s in p["steps"]:
        print(indent + f"    {s['N']:>5} -> {s['N2']:<5} "
                       f"Delta = {s['delta']:+.5f} +- {s['sem']:.5f}  "
                       f"95% [{s['ci'][0]:+.5f}, {s['ci'][1]:+.5f}]  "
                       f"HW/tau_I = {s['half_width_over_tau']:.2f}  "
                       f"P1={'y' if s['P1_compatible_with_zero'] else 'n'} "
                       f"P2={'y' if s['P2_ci_inside_tau_I'] else 'n'}  "
                       f"B_eff = {s['B_eff']:+.3f} +- {s['B_eff_sem']:.3f}  "
                       f"R for P2 = {s['R_required_for_P2']}")
    print(indent + f"    criteria {p.get('criteria')}")
    print(indent + f"    VERDICT  {p.get('verdict')}"
          + (f"   (binding budget: {p['binding_budget']})"
             if p.get("binding_budget") else ""))
    a = r["asymptotic"]
    if a.get("status") == "insufficient":
        print(indent + f"    asymptotic form: {a['reason']}")
        return
    f = a["fit_1overN"]
    print(indent + f"    1/N fit: I_inf = {f['I_inf']:.5f} +- {f['se_I']:.5f}, "
                   f"B = {f['B']:+.3f} +- {f['se_B']:.3f}, "
                   f"chi2 = {f['chi2']:.2f} / {f['dof']} dof"
          + (f", p = {f['p']:.4f}" if f["p"] is not None else ", no dof"))
    g = a["fit_free_gamma"]
    if g:
        print(indent + f"    free-gamma fit: gamma = {g['gamma']:.3f}, "
                       f"chi2 = {g['chi2']:.2f} / {g['dof']} dof"
              + ("   (3 parameters on 3 rungs — exact, not a test)"
                 if g["dof"] <= 0 else ""))
    print(indent + f"    B_eff stable on top two steps: "
                   f"{a['B_eff_stable_on_top_two']}")
    print(indent + f"    ASYMPTOTIC VERDICT  {a['verdict']}")
    if a.get("coefficient_B"):
        print(indent + f"      B = {a['coefficient_B']['B']:+.4f} "
                       f"+- {a['coefficient_B']['se']:.4f}   "
                       f"{a['coefficient_B']['scope']}")
    else:
        print(indent + f"      no coefficient quoted: {a.get('why')}")


def shape_tests(B):
    """H1 additive constant, H2 multiplicative rescaling, H3 resolved
    lambda-dependent shape distortion. Seven points is not many; each
    hypothesis costs ONE parameter and the residual chi2 has 6 dof."""
    out = {}
    for lo, hi in ((512, 1024), (1024, 2048)):
        pts = [(l, a, b) for l, a, b in zip(D.B_GRID, B.get(lo, []), B.get(hi, []))
               if a is not None and b is not None and a.R > 1 and b.R > 1]
        key = f"{lo}->{hi}"
        if len(pts) < 4:
            out[key] = dict(status="insufficient", n=len(pts))
            continue
        lams = [p[0] for p in pts]
        d = np.array([delta(p[1], p[2])[0] for p in pts])
        s = np.array([delta(p[1], p[2])[1] for p in pts])
        lo_m = np.array([p[1].mean for p in pts])
        lo_e = np.array([p[1].sem for p in pts])
        w = 1.0 / s ** 2
        # H1: Delta(lambda) = const
        c1 = float((w * d).sum() / w.sum())
        chi1 = float((w * (d - c1) ** 2).sum())
        # H2: I_hi = (1+eps) I_lo, i.e. Delta = eps * I_lo
        wr = 1.0 / (s ** 2 + (c1 / np.maximum(lo_m, 1e-9)) ** 2 * lo_e ** 2)
        eps = float((wr * d * lo_m).sum() / (wr * lo_m ** 2).sum())
        chi2v = float((wr * (d - eps * lo_m) ** 2).sum())
        dof = len(pts) - 1
        # H3: is there resolved lambda dependence at all?
        chi0 = float((w * (d - d.mean()) ** 2).sum())
        out[key] = dict(
            status="ok", lams=lams, delta=[float(v) for v in d],
            delta_sem=[float(v) for v in s],
            H1_additive=dict(constant=c1, chi2=chi1, dof=dof,
                             p=chi2_sf(chi1, dof), rejected=bool(
                                 chi2_sf(chi1, dof) < 0.05)),
            H2_multiplicative=dict(epsilon=eps, chi2=chi2v, dof=dof,
                                   p=chi2_sf(chi2v, dof), rejected=bool(
                                       chi2_sf(chi2v, dof) < 0.05)),
            H3_shape_resolved=bool(chi2_sf(chi0, len(pts) - 1) < 0.05),
            ratio_centred=[float(v) for v in (d - eps * lo_m)],
            note="Seven points. Neither H1 nor H2 is 'accepted' by failing to "
                 "be rejected; both surviving is an UNRESOLVED outcome and is "
                 "reported as one.")
    return out


def report_shape(sh):
    for k, v in sh.items():
        if v.get("status") != "ok":
            print(f"  {k}: insufficient ({v['n']} usable lambdas)")
            continue
        h1, h2 = v["H1_additive"], v["H2_multiplicative"]
        print(f"  {k}:")
        print(f"    H1 additive constant  {h1['constant']:+.5f}   "
              f"chi2 {h1['chi2']:.2f}/{h1['dof']} p={h1['p']:.4f}  "
              f"{'REJECTED' if h1['rejected'] else 'not rejected'}")
        print(f"    H2 multiplicative eps {h2['epsilon']:+.5f}   "
              f"chi2 {h2['chi2']:.2f}/{h2['dof']} p={h2['p']:.4f}  "
              f"{'REJECTED' if h2['rejected'] else 'not rejected'}")
        print(f"    H3 resolved lambda-dependent shape distortion: "
              f"{v['H3_shape_resolved']}")
        if not h1["rejected"] and not h2["rejected"]:
            print("    -> UNRESOLVED: seven points do not separate an additive "
                  "displacement from a multiplicative one here.")


def locator(cells):
    """Section 4B. Two diagnostics, because only one of them is always
    constructible:

    (a) FULLY MATCHED — both curves at the same N_c. Needs the low-L reference
        at that N_c, which campaign B2 exists to supply.
    (b) ONE-SIDED — the L = 64 curve moves in N_c against a reference held at
        N_c = 1024. This isolates the L = 64 side of the displacement, which is
        exactly the part that does NOT cancel in a cross-L difference.

    Both are reported. A displacement common to both L cancels in D and does not
    move the crossing; only the L-DEPENDENT part does. That distinction is the
    load-bearing one and it is why absolute-level convergence and locator
    convergence are separate questions.
    """
    out = dict(pairs={}, matched={}, one_sided={})
    grid = D.B2_GRID          # the lambdas shared by B and the measured corpus

    def cv(L, N):
        cs = curve(cells, L, N, grid)
        if any(c is None or c.R < 2 for c in cs):
            return None
        return ([c.mean for c in cs], [c.sem for c in cs])

    for (La, Lb) in ((32, 64), (48, 64), (32, 48)):
        for N in (512, 1024, 2048):
            a, b = cv(La, N), cv(Lb, N)
            if a and b:
                r = crossings(grid, a[0], a[1], b[0], b[1])
                r["status"] = "ok"
                r["loo_lambda"] = loo_lambda_crossing(grid, a[0], a[1],
                                                      b[0], b[1])
                out["matched"][f"L{La}-L{Lb}@Nc{N}"] = r
                out["pairs"][f"L{La}-L{Lb} $N_c$={N}"] = r
        ref = cv(La, 1024)
        if ref:
            for N in (512, 1024, 2048, 4096, 8192):
                b = cv(Lb, N)
                if b:
                    r = crossings(grid, ref[0], ref[1], b[0], b[1])
                    r["status"] = "ok"
                    out["one_sided"][f"L{La}@1024 - L{Lb}@{N}"] = r
    # how far the crossing moves per doubling, against tau_lambda
    mv = {}
    for base in ("matched", "one_sided"):
        ks = sorted(out[base])
        for k1, k2 in itertools.combinations(ks, 2):
            r1, r2 = out[base][k1], out[base][k2]
            if r1["bootstrap_median"] is None or r2["bootstrap_median"] is None:
                continue
            d = r2["bootstrap_median"] - r1["bootstrap_median"]
            mv[f"{base}: {k1} -> {k2}"] = dict(
                shift=d, abs_shift=abs(d),
                within_tau_lambda=bool(abs(d) <= D.TAU_LAMBDA),
                tau_lambda=D.TAU_LAMBDA)
    out["crossing_shifts"] = mv
    out["note"] = ("|dD/dlambda| measured on the existing N_c=1024 curves at "
                   f"the interior crossings is >= {D.DDDLAM_MIN}, which is what "
                   f"turns tau_lambda={D.TAU_LAMBDA} into tau_D={D.TAU_D}. If "
                   "the measured slope in the new data differs materially, the "
                   "translation must be redone before any adequacy verdict.")
    return out


def report_locator(lo):
    if not lo["matched"] and not lo["one_sided"]:
        print("  no pair is constructible yet.")
        print("  Fully matched comparisons need the L=32/48 reference at the "
              "same N_c\n  (campaign B2). Without B2 only the one-sided "
              "diagnostic exists, and\n  that limitation would have to be "
              "stated rather than worked around.")
        return
    for base in ("matched", "one_sided"):
        for k, r in sorted(lo[base].items()):
            print(f"  [{base}] {k}")
            print(f"      raw sign changes {r['raw_sign_changes']}, resolved "
                  f"{r['resolved_sign_changes']}, endpoint-induced "
                  f"{r['endpoint_induced']}")
            print(f"      crossing(s) {['%.5f' % v for v in r['crossing_lambdas']]}"
                  f"  bootstrap median "
                  f"{r['bootstrap_median'] if r['bootstrap_median'] is None else round(r['bootstrap_median'], 5)}"
                  f"  95% {r['bootstrap_ci']}")
            print(f"      fraction of bootstrap replicates with exactly one "
                  f"crossing {r['fraction_exactly_one']:.3f}")
    for k, v in sorted(lo["crossing_shifts"].items()):
        print(f"  shift {k}: {v['shift']:+.5f}  "
              f"{'WITHIN' if v['within_tau_lambda'] else 'OUTSIDE'} "
              f"tau_lambda = {v['tau_lambda']}")


def discretisation(cells):
    """E1 window-count effect vs E2 discretisation-stable limit. Frozen before
    any datum: K-accumulation predicts drift proportional to 1/dtau_mult;
    a schedule-independent mechanism predicts approximate flatness. An
    intermediate result kills neither and is reported INCONCLUSIVE."""
    out = {}
    for N in D.E_NCS:
        pts = []
        for dm in D.E_DTAUS:
            c = cells.get((64, 64.0, 0.35, round(D.E_LAM, 6), N, dm))
            if c is not None and c.R > 1:
                pts.append((dm, c))
        if len(pts) < 3:
            out[str(N)] = dict(status="insufficient", have=len(pts))
            continue
        dms = np.array([p[0] for p in pts])
        y = np.array([p[1].mean for p in pts])
        e = np.array([p[1].sem for p in pts])
        w = 1.0 / e ** 2
        c0 = float((w * y).sum() / w.sum())
        chi_flat = float((w * (y - c0) ** 2).sum())
        x = 1.0 / dms
        S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
        Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
        den = S * Sxx - Sx * Sx
        b = (S * Sxy - Sx * Sy) / den
        a0 = (Sxx * Sy - Sx * Sxy) / den
        chi_k = float((w * (y - (a0 + b * x)) ** 2).sum())
        p_flat = chi2_sf(chi_flat, len(pts) - 1)
        p_k = chi2_sf(chi_k, len(pts) - 2)
        if p_flat >= 0.05 and (b == 0 or abs(b) <= Z95 * math.sqrt(S / den)):
            v = "E2_DISCRETISATION_STABLE"
        elif p_flat < 0.05 and p_k >= 0.05:
            v = "E1_WINDOW_COUNT_EFFECT"
        else:
            v = "INCONCLUSIVE"
        out[str(N)] = dict(
            status="ok", dtau_mult=[float(v2) for v2 in dms],
            K=[int(p[1].pops[0]["n_steps"]) for p in pts],
            mean=[float(v2) for v2 in y], sem=[float(v2) for v2 in e],
            R=[p[1].R for p in pts],
            E2_flat=dict(constant=c0, chi2=chi_flat, dof=len(pts) - 1, p=p_flat),
            E1_one_over_dtau=dict(intercept=a0, slope=b, se_slope=math.sqrt(S / den),
                                  chi2=chi_k, dof=len(pts) - 2, p=p_k),
            verdict=v,
            note="dtau_mult is a DISCRETISATION CONTROL, not a physical "
                 "parameter, and K is not called the causal variable unless "
                 "E1 is supported. The dtau_mult != 6 rows may never be pooled "
                 "with the production corpus.")
    if all(v.get("status") == "ok" for v in out.values()) and out:
        vs = {k: v["verdict"] for k, v in out.items()}
        out["joint"] = dict(
            per_Nc=vs,
            agree=len(set(vs.values())) == 1,
            note="The K-dependence may itself depend on population size, which "
                 "is why both N_c are reported and neither is pooled into the "
                 "other.")
    return out


def report_E(e):
    for k, v in sorted(e.items()):
        if k == "joint":
            print(f"  joint: {v['per_Nc']}  (agree: {v['agree']})")
            continue
        if v.get("status") != "ok":
            print(f"  N_c={k}: insufficient ({v['have']} of 3 dtau_mult present)")
            continue
        print(f"  N_c={k}:  K = {v['K']}  dtau_mult = {v['dtau_mult']}")
        for dm, m, s, R in zip(v["dtau_mult"], v["mean"], v["sem"], v["R"]):
            print(f"      dtau_mult={dm:<5g} R={R:<3} CMI = {m:.5f} +- {s:.5f}")
        print(f"      E2 flat:         chi2 {v['E2_flat']['chi2']:.2f}/"
              f"{v['E2_flat']['dof']}  p={v['E2_flat']['p']:.4f}")
        print(f"      E1 ~ 1/dtau_mult: slope "
              f"{v['E1_one_over_dtau']['slope']:+.5f} +- "
              f"{v['E1_one_over_dtau']['se_slope']:.5f}  "
              f"chi2 {v['E1_one_over_dtau']['chi2']:.2f}/"
              f"{v['E1_one_over_dtau']['dof']}  p={v['E1_one_over_dtau']['p']:.4f}")
        print(f"      VERDICT {v['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
