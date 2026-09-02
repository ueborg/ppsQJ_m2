#!/usr/bin/env python3
"""The ONE place TASK-2026-09-02-MOCK-PRODUCTION's frozen criteria are evaluated.

Implements brief sections 9 (figures), 10 (curve quality), 11 (crossings),
12 (high-N_c vs low-N_c) and 13 (M1-M7), under the rules frozen in
../analysis_spec.yaml. Nothing here is decided after the data arrive.

Every uncertainty is ACROSS INDEPENDENT POPULATIONS. Within-clone spread appears
only as VIF/N_eff and is never a standard error.

MATCHED-R AMENDMENT (see ../MATCHED_R_AMENDMENT.md)
---------------------------------------------------
The PRIMARY comparison of curve quality between N_c classes is matched at
R = 24 independent populations per (L, lambda) cell, so that

    "the N_c=1024 curve is cleaner than the N_c=128 curve"

can never be an artefact of unequal R. Cells that hold more than 24 populations
are cut into consecutive disjoint blocks of 24 in SEED order -- a rule that is
deterministic and observable-blind, fixed before any datum is seen:

    reused ARM-B cells   R=96  ->  blocks A B C D   (24 each)
    N_c=128 comparator   R=48  ->  blocks A B       (24 each)
    everything else      R=24  ->  block A only

Block A is always the primary dataset. B/C/D are independent
sensitivity/replication checks and are NEVER an opportunity to pick the
nicest-looking subset. The full-R estimates are SECONDARY high-precision means
and may not carry any primary curve-quality or crossing statistic.

A consequence worth stating: the primary analysis is uniformly R = 24 at every
cell of every curve, so the per-point error bars are homoscedastic by
construction and the earlier unequal-R caveat no longer applies to it.

Runs to completion with zero new results and degrades explicitly.
Contains no scheduler call. Reads only; writes only inside this task directory.

    python3 analysis/mock_production_analysis.py
"""
import os, sys, csv, json, glob, math, itertools, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
FIGDIR = os.path.join(HERE, "figures")

# ----- frozen constants, mirrored from analysis_spec.yaml -------------------
GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
DLAM = 0.010
ZETA = 0.35
DTAU_MULT = 6.0
LS = [32, 48, 64]
PAIRS = [(32, 48), (32, 64), (48, 64)]
NC_MAIN, NC_HIGH, NC_LOW = 1024, 2048, 128
CENTRE3 = [0.2932, 0.3032, 0.3132]
B_BOOT = 10000
SEED = 20260902
ARMS = ["mockL32", "mockL48", "mockL64", "mockL64nc2048",
        "mockNC128L32", "mockNC128L48", "mockNC128L64"]

# --- the matched-R amendment ----------------------------------------------
# BLOCK is the matched number of independent populations per (L, lambda) cell
# for every PRIMARY statistic. Cells with more are cut into consecutive
# disjoint blocks of this size in SEED order. BLOCK_LABELS[k] names them.
BLOCK = 24
BLOCK_LABELS = "ABCDEFGH"
PRIMARY_BLOCK = 0                     # block A, always
HIST_PRECISION_R = 12                 # the historical corpus's own R

# Audit fields for M6. They are constants because this script never smooths and
# never removes a lambda point; they are WRITTEN OUT so that a future edit that
# changed that would have to change them too.
AUDIT = dict(smoothing_applied=False, value_based_exclusions=0,
             lambda_points_removed=0)


# ===========================================================================
# loading
# ===========================================================================
def _cellkey(L, N_c, lam):
    return (int(L), int(N_c), round(float(lam), 6))


def load():
    """Return cells[(L, N_c, lam)] = dict(pops=[...], within=[...], meta...)."""
    cells, excluded, suspect, nonok = {}, [], [], 0

    def add(L, N_c, lam, rec):
        c = cells.setdefault(_cellkey(L, N_c, lam),
                             dict(pops=[], within=[], nonfin=0, clones=0,
                                  fallbacks=0, anc=[], wall=[], seeds=[],
                                  src=set()))
        c["pops"].append(rec["mean"])
        c["within"].append(rec["within"])
        c["nonfin"] += rec["nonfin"]
        c["clones"] += int(N_c)
        c["fallbacks"] += rec["fallbacks"]
        c["anc"].append(rec["anc"])
        c["wall"].append(rec["wall"])
        c["seeds"].append(rec["seed"])
        c["src"].add(rec["src"])

    # frozen reused ARM-B populations
    fp = os.path.join(TASK, "frozen_inputs", "armB_populations.csv")
    n_frozen = 0
    if os.path.isfile(fp):
        for r in csv.DictReader(open(fp)):
            if r["status"] != "ok":
                continue
            add(r["L"], r["N_c"], r["lam"],
                dict(mean=float(r["cmi_weighted_mean"]),
                     within=float(r["cmi_within_var"]),
                     nonfin=int(r["n_nonfinite"]),
                     fallbacks=int(r["brentq_fallbacks"]),
                     anc=int(r["n_distinct_anc_final"]),
                     wall=float(r["wall_s"]),
                     seed=int(r["seed"]),
                     src="frozen:ARM-B"))
            n_frozen += 1

    n_new = 0
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(TASK, arm, "results", "*.json"))):
            d = json.load(open(p))
            if d.get("status") not in (None, "ok"):
                nonok += 1
                continue
            add(d["L"], d["N_c"], d["lam"],
                dict(mean=float(d["cmi_weighted_mean"]),
                     within=float(d["cmi_within_var"]),
                     nonfin=int(d.get("n_nonfinite", 0)),
                     fallbacks=int(d.get("brentq_fallbacks", 0)),
                     anc=int(d.get("n_distinct_anc_final", 0)),
                     wall=float(d.get("wall_s", float("nan"))),
                     seed=int(d["seed"]),
                     src=arm))
            n_new += 1

    for k, c in cells.items():
        # SEED ORDER, fixed here and nowhere else. Every block cut downstream
        # is a slice of this ordering, so block membership depends only on the
        # seeds -- never on CMI, and never on the order files happened to be
        # read off the filesystem.
        order = np.argsort(np.asarray(c["seeds"], dtype=np.int64), kind="stable")
        for f in ("pops", "within", "anc", "wall", "seeds"):
            c[f] = np.asarray(c[f], float if f != "seeds" else np.int64)[order]
        c["N_c"] = int(k[1])
        c["lam"] = float(k[2])
        c.update(_stats(c["pops"], c["within"], c["N_c"]))
        c["n_blocks"] = c["R"] // BLOCK
        c["block"] = None                      # None == "the whole cell"
        if c["nonfin"] > 0.01 * c["clones"]:
            suspect.append(k)
    return cells, dict(n_frozen=n_frozen, n_new=n_new, nonok=nonok,
                       suspect=suspect, excluded=excluded)


def _stats(pops, within, N_c):
    """Across-population statistics for a population set. The ONLY error bar."""
    R = int(pops.size)
    var = float(pops.var(ddof=1)) if R > 1 else float("nan")
    wm = float(np.mean(within)) if within.size else float("nan")
    return dict(R=R, mean=float(pops.mean()) if R else float("nan"), var=var,
                sem=math.sqrt(var / R) if R > 1 else float("nan"),
                vif=var * N_c / wm if wm and wm > 0 else float("nan"),
                n_eff=wm / var if var and var > 0 else float("nan"))


def cell_block(c, k, size=BLOCK):
    """Block k of a cell: populations [k*size, (k+1)*size) in SEED order.

    Deterministic and OBSERVABLE-BLIND. The ordering was fixed in load() by
    argsort over the seeds alone, so permuting the CMI values within a cell
    cannot change which population lands in which block. That property is
    asserted by the block-selection unit check in ../VALIDATION.md.

    Returns None if the cell does not hold a full block k.
    """
    lo, hi = k * size, (k + 1) * size
    if hi > int(c["R"]):
        return None
    sub = dict(c)
    for f in ("pops", "within", "anc", "wall", "seeds"):
        sub[f] = c[f][lo:hi]
    sub.update(_stats(sub["pops"], sub["within"], c["N_c"]))
    sub["block"] = k
    sub["block_label"] = BLOCK_LABELS[k]
    sub["parent_R"] = int(c["R"])
    # nonfin/clones/fallbacks stay PARENT-level: the exclusion accounting is a
    # property of the cell as run, and the SUSPECT flag is raised there. They
    # are not re-scaled to the block, and nothing downstream reads them as if
    # they were.
    sub["parent_level_fields"] = ("nonfin", "clones", "fallbacks")
    return sub


def curve(cells, L, N_c, block=PRIMARY_BLOCK):
    """The 13-point curve at matched R = BLOCK, or None if a point is missing.

    `block=None` returns the FULL cells (all R). That is the SECONDARY
    high-precision view and must not carry a primary curve-quality or crossing
    statistic -- see ../MATCHED_R_AMENDMENT.md.
    """
    ks = [_cellkey(L, N_c, l) for l in GRID]
    if any(k not in cells for k in ks):
        return None
    if block is None:
        return None if any(cells[k]["R"] < 2 for k in ks) else [cells[k] for k in ks]
    out = [cell_block(cells[k], block) for k in ks]
    return None if any(o is None for o in out) else out


def n_blocks_available(cells, L, N_c):
    """How many full matched-R blocks every point of this curve can supply."""
    ks = [_cellkey(L, N_c, l) for l in GRID]
    if any(k not in cells for k in ks):
        return 0
    return min(int(cells[k]["n_blocks"]) for k in ks)


# ===========================================================================
# bootstrap machinery — resample INDEPENDENT POPULATIONS within each cell
# ===========================================================================
def boot_curves(cs, rng, B):
    """B x 13 array of bootstrap curve means."""
    out = np.empty((B, len(cs)))
    for j, c in enumerate(cs):
        p = c["pops"]
        idx = rng.integers(0, p.size, size=(B, p.size))
        out[:, j] = p[idx].mean(axis=1)
    return out


# ===========================================================================
# section 10 — curve quality
# ===========================================================================
def curve_quality(cs, rng):
    m = np.array([c["mean"] for c in cs])
    s = np.array([c["sem"] for c in cs])
    d = np.diff(m)
    sd = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    r = np.abs(d) / sd
    q = m[2:] - 2 * m[1:-1] + m[:-2]
    sq = np.sqrt(s[:-2] ** 2 + 4 * s[1:-1] ** 2 + s[2:] ** 2)
    rough = float(np.mean((q / sq) ** 2))

    bc = boot_curves(cs, rng, B_BOOT)
    bq = bc[:, 2:] - 2 * bc[:, 1:-1] + bc[:, :-2]
    brough = np.mean((bq / sq) ** 2, axis=1)

    # weighted quadratic yardstick -- a comparator, never a replacement
    x = np.array(GRID) - float(np.mean(GRID))
    W = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    coef, *_ = np.linalg.lstsq(A * np.sqrt(W)[:, None], m * np.sqrt(W), rcond=None)
    chi2 = float(np.sum(W * (A @ coef - m) ** 2))
    return dict(mean=m.tolist(), sem=s.tolist(),
                d=d.tolist(), sem_d=sd.tolist(), r=r.tolist(),
                r_median=float(np.median(r)), n_r_ge_2=int((r >= 2).sum()),
                q=q.tolist(), sem_q=sq.tolist(),
                roughness=rough,
                roughness_ci=[float(np.percentile(brough, 2.5)),
                              float(np.percentile(brough, 97.5))],
                quad_chi2=chi2, quad_chi2_dof=chi2 / (len(GRID) - 3),
                quad_coef=coef.tolist())


def population_diagnostics(cs, rng):
    """Q5 split-half, Q7 outlier and leave-one-out, per cell."""
    out = []
    for c, lam in zip(cs, GRID):
        p = c["pops"]
        R = p.size
        perm = np.random.default_rng(SEED).permutation(R)
        a, b = p[perm[:R // 2]], p[perm[R // 2:2 * (R // 2)]]
        if a.size >= 2 and b.size >= 2:
            dsh = float(a.mean() - b.mean())
            ssh = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            sh_ok = abs(dsh) <= 2.5 * ssh
        else:
            dsh, ssh, sh_ok = float("nan"), float("nan"), None
        sd = p.std(ddof=1)
        z = np.abs(p - p.mean()) / sd if sd > 0 else np.zeros_like(p)
        zmax = float(z.max())
        zthr = float(_z_for_R(R))
        i = int(np.argmax(z))
        loo = np.delete(p, i).mean()
        shift = abs(loo - p.mean()) / c["sem"] if c["sem"] > 0 else float("inf")
        out.append(dict(lam=lam, R=R, split_half=dsh, split_half_sem=ssh,
                        split_half_ok=sh_ok, zmax=zmax, z_thr=zthr,
                        outlier_ok=zmax <= zthr,
                        loo_shift_sem=float(shift), loo_ok=shift <= 1.0))
    return out


def _z_for_R(R):
    """Two-sided 1 % expected-maximum threshold for R draws."""
    from math import erf, sqrt
    p = 1 - 0.01 / (2 * R)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# section 11 — crossing protocol
# ===========================================================================
def _crossings_of(D, lams):
    """All raw sign changes with their interpolated lambdas."""
    out = []
    for i in range(len(D) - 1):
        if D[i] == 0.0:
            out.append((i, lams[i]))
        elif D[i] * D[i + 1] < 0:
            t = D[i] / (D[i] - D[i + 1])
            out.append((i, lams[i] + t * (lams[i + 1] - lams[i])))
    return out


def crossing_analysis(cA, cB, rng, label):
    """cA is the smaller L. D = I_B - I_A."""
    mA = np.array([c["mean"] for c in cA]); sA = np.array([c["sem"] for c in cA])
    mB = np.array([c["mean"] for c in cB]); sB = np.array([c["sem"] for c in cB])
    D = mB - mA
    sD = np.sqrt(sA ** 2 + sB ** 2)
    raw = _crossings_of(D.tolist(), GRID)
    resolved = [(i, x) for (i, x) in raw
                if abs(D[i]) >= 2 * sD[i] and abs(D[i + 1]) >= 2 * sD[i + 1]]

    bA = boot_curves(cA, rng, B_BOOT)
    bB = boot_curves(cB, rng, B_BOOT)
    bD = bB - bA
    counts = collections.Counter()
    firsts = []
    for row in bD:
        cs = _crossings_of(row.tolist(), GRID)
        counts[len(cs)] += 1
        if len(cs) == 1:
            firsts.append(cs[0][1])
    frac_one = counts[1] / B_BOOT
    if firsts:
        ci = [float(np.percentile(firsts, 2.5)), float(np.percentile(firsts, 97.5))]
    else:
        ci = [float("nan"), float("nan")]

    unique = (len(raw) == 1) and (frac_one >= 0.95)
    endpoint = any(i == 0 or i == len(GRID) - 2 or
                   x <= GRID[0] + DLAM / 2 or x >= GRID[-1] - DLAM / 2
                   for i, x in raw)
    width = ci[1] - ci[0] if firsts else float("nan")
    boot_stable = bool(firsts) and width <= 2 * DLAM

    # jackknife over deleted lambda points -- a STABILITY DIAGNOSTIC ONLY
    jack = []
    for k in range(len(GRID)):
        idx = [j for j in range(len(GRID)) if j != k]
        cs = _crossings_of(D[idx].tolist(), [GRID[j] for j in idx])
        jack.append(cs[0][1] if len(cs) == 1 else None)
    jack_ok = (bool(firsts)
               and all(v is not None and ci[0] <= v <= ci[1] for v in jack))

    return dict(pair=label, D=D.tolist(), sem_D=sD.tolist(),
                n_raw=len(raw), n_resolved=len(resolved),
                crossings=[x for _, x in raw],
                resolved_crossings=[x for _, x in resolved],
                boot_ci=ci, boot_width=width,
                boot_count_hist={str(k): v for k, v in sorted(counts.items())},
                frac_exactly_one=frac_one,
                unique=unique, endpoint_induced=endpoint,
                stable_bootstrap=boot_stable,
                jackknife=[None if v is None else float(v) for v in jack],
                stable_point_deletion=jack_ok)


# ===========================================================================
# section 12 — high-N_c versus low-N_c
# ===========================================================================
def nc_comparison(hi, lo, L):
    mh = np.array([c["mean"] for c in hi]); sh = np.array([c["sem"] for c in hi])
    ml = np.array([c["mean"] for c in lo]); sl = np.array([c["sem"] for c in lo])
    D = mh - ml
    sD = np.sqrt(sh ** 2 + sl ** 2)
    x = np.array(GRID) - float(np.mean(GRID))
    W = 1.0 / sD ** 2

    def fit(deg):
        A = np.vstack([x ** k for k in range(deg + 1)]).T
        c, *_ = np.linalg.lstsq(A * np.sqrt(W)[:, None], D * np.sqrt(W), rcond=None)
        return c, float(np.sum(W * (A @ c - D) ** 2))

    c0, x0 = fit(0)
    c1, x1 = fit(1)
    c2, x2 = fit(2)
    n = len(GRID)
    if x0 / (n - 1) <= 1.5:
        cls = "predominantly a vertical shift"
    elif x2 / (n - 3) > 2.0:
        cls = "irregular pointwise displacement"
    elif (x1 - x2) > 9:
        cls = "curvature change"
    elif (x0 - x1) > 9:
        cls = "slope change"
    else:
        cls = "unclassified: no single description dominates"
    return dict(L=L, delta=D.tolist(), sem_delta=sD.tolist(),
                mean_shift=float(c0[0]), mean_shift_sem=float(1 / math.sqrt(W.sum())),
                chi2_const=x0, chi2_const_dof=x0 / (n - 1),
                chi2_linear=x1, chi2_linear_dof=x1 / (n - 2),
                chi2_quad=x2, chi2_quad_dof=x2 / (n - 3),
                classification=cls)


# ===========================================================================
# figures
# ===========================================================================
def figures(cells, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  figures SKIPPED: matplotlib unavailable ({e})")
        return []
    os.makedirs(FIGDIR, exist_ok=True)
    made = []
    COL = {32: "#1b6ca8", 48: "#c85200", 64: "#118050"}

    def _curveplot(ax, N_c, title):
        any_ = False
        for L in LS:
            cs = curve(cells, L, N_c, block=PRIMARY_BLOCK)
            if cs is None:
                continue
            any_ = True
            m = [c["mean"] for c in cs]
            s = [c["sem"] for c in cs]
            ax.errorbar(GRID, m, yerr=s, marker="o", ms=4, lw=1.2, capsize=2,
                        color=COL[L], label=f"L = {L} (matched R = {BLOCK})")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("CMI")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25)
        if any_:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "no complete curve yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
        return any_

    # ---- FIGURE A
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _curveplot(ax, NC_MAIN, r"A  —  CMI$(\lambda)$, $N_c=1024$, $\zeta=0.35$, "
                            r"$T=L$   (no smoothing)")
    fig.tight_layout(); p = os.path.join(FIGDIR, "figureA_cmi_nc1024.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # ---- FIGURE B : matched companion (primary) + historical corpus (descriptive)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    _curveplot(axes[0], NC_LOW, r"B1  —  matched $N_c=128$ companion "
                                r"($d\tau_{\rm mult}=6$, exact common cells)")
    ax = axes[1]
    hp = os.path.join(TASK, "frozen_inputs", "historical_corpus_zeta035.csv")
    if os.path.isfile(hp):
        g = collections.defaultdict(list)
        for r in csv.DictReader(open(hp)):
            g[(int(r["L"]), float(r["lambda"]))].append(float(r["CMI_mean"]))
        for L in (64,):
            xs = sorted(l for (LL, l) in g if LL == L and 0.20 <= l <= 0.40)
            m = [float(np.mean(g[(L, l)])) for l in xs]
            s = [float(np.std(g[(L, l)], ddof=1) / math.sqrt(len(g[(L, l)]))) for l in xs]
            ax.errorbar(xs, m, yerr=s, marker="s", ms=4, lw=1.2, ls="--",
                        capsize=2, color=COL[L], label=f"L = {L}, R = 12")
        for l in GRID:
            ax.axvline(l, color="#ccc", lw=.5, zorder=0)
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("CMI")
    ax.set_title(r"B2  —  historical corpus, $N_c=128$, "
                 r"$d\tau_{\rm mult}=12$", fontsize=10)
    ax.text(.97, .93, "DESCRIPTIVE ONLY\nno exact common cell\nwith this campaign",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#a00", bbox=dict(fc="white", ec="#a00", alpha=.85))
    ax.grid(alpha=.25)
    fig.tight_layout(); p = os.path.join(FIGDIR, "figureB_lowNc_comparison.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # ---- FIGURE C
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    got = False
    for e in res.get("nc_comparison", []):
        got = True
        ax.errorbar(GRID, e["delta"], yerr=e["sem_delta"], marker="o", ms=4,
                    lw=1.2, capsize=2, color=COL[e["L"]],
                    label=f"L = {e['L']}: {e['classification']}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$I_{N_c=1024} - I_{N_c=128}$")
    ax.set_title("C  —  high-$N_c$ minus low-$N_c$, exact common cells only",
                 fontsize=10)
    ax.grid(alpha=.25)
    if got:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "no exact common (N_c=1024, N_c=128) cells yet.\n"
                        "The historical corpus is NOT interpolated to fill this.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#a00")
    fig.tight_layout(); p = os.path.join(FIGDIR, "figureC_nc_difference.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # ---- FIGURE D
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ok = True
    for N_c, col, mk in ((NC_MAIN, "#118050", "o"), (NC_HIGH, "#7b3fbf", "D")):
        ks = [_cellkey(64, N_c, l) for l in CENTRE3]
        if any(k not in cells for k in ks):
            ok = False
            continue
        # matched R = BLOCK on both sides, exactly as the primary Delta_N
        bl = [cell_block(cells[k], PRIMARY_BLOCK) for k in ks]
        if any(b is None for b in bl):
            ok = False
            continue
        m = [b["mean"] for b in bl]; s = [b["sem"] for b in bl]
        axes[0].errorbar(CENTRE3, m, yerr=s, marker=mk, ms=5, lw=1.2, capsize=3,
                         color=col,
                         label=f"$N_c$ = {N_c} (matched R = {BLOCK})")
    dn = res.get("delta_N")
    if dn:
        axes[1].errorbar(CENTRE3, dn["delta"], yerr=dn["sem"], marker="o", ms=5,
                         lw=1.2, capsize=3, color="#7b3fbf")
        axes[1].axhline(0, color="k", lw=.8)
    axes[0].set_ylabel("CMI"); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
    axes[0].set_title(r"D  —  $L=64$: $N_c=1024$ vs $N_c=2048$ at the three "
                      r"common $\lambda$", fontsize=10)
    axes[1].set_ylabel(r"$\Delta_N(\lambda)$"); axes[1].set_xlabel(r"$\lambda$")
    axes[1].grid(alpha=.25)
    if not ok or not dn:
        axes[1].text(.5, .5, "N_c = 2048 arm incomplete", ha="center",
                     va="center", transform=axes[1].transAxes, color="#a00")
    fig.tight_layout(); p = os.path.join(FIGDIR, "figureD_nc2048_shapecheck.png")
    fig.savefig(p, dpi=160); plt.close(fig); made.append(p)
    return made


# ===========================================================================
# main
# ===========================================================================
def hdr(t):
    print("\n" + "=" * 78); print(f"  {t}"); print("=" * 78)


def main():
    rng = np.random.default_rng(SEED)
    cells, meta = load()
    res = dict(audit=AUDIT, loaded=meta)

    print("=" * 78)
    print("  TASK-2026-09-02-MOCK-PRODUCTION — mock-production analysis")
    print("  frozen rules: analysis_spec.yaml (sections 9-13 of the brief)")
    print("=" * 78)
    print(f"  populations loaded: {meta['n_new'] + meta['n_frozen']} "
          f"({meta['n_new']} new, {meta['n_frozen']} frozen ARM-B)")
    print(f"  cells: {len(cells)}   runs with status != ok: {meta['nonok']}")
    if meta["suspect"]:
        print(f"  ** SUSPECT cells (>1 % non-finite clones): {meta['suspect']}")
    print("  every error bar below is ACROSS INDEPENDENT POPULATIONS; within-clone")
    print("  spread appears only as VIF/N_eff and is never a standard error.")
    print()
    print("  MATCHED-R AMENDMENT (../MATCHED_R_AMENDMENT.md)")
    print(f"  PRIMARY: every curve, crossing and quality statistic below is")
    print(f"  computed at a MATCHED R = {BLOCK} per (L, lambda) cell, taken as")
    print(f"  block A -- the first {BLOCK} populations in SEED order. Block")
    print("  membership is observable-blind: it cannot depend on CMI.")
    print("  SECONDARY: full-R means, replicate blocks B/C/D, and R=12")
    print("  historical-precision checks are reported separately and have NO")
    print("  authority over any 'cleaner curve' statement.")

    # blocks available per curve -- the audit trail for the matched-R claim
    hdr("A0.  MATCHED-R BLOCK INVENTORY")
    res["block_inventory"] = {}
    print(f"  {'curve':<22}{'R per cell':>12}{'full blocks':>13}"
          f"{'primary R':>11}   seed-ordered block boundaries (lambda_0)")
    for N_c in (NC_MAIN, NC_HIGH, NC_LOW):
        for L in LS:
            ks = [_cellkey(L, N_c, l) for l in GRID]
            present = [k for k in ks if k in cells]
            if not present:
                continue
            Rs = sorted({int(cells[k]["R"]) for k in present})
            nb = min(int(cells[k]["n_blocks"]) for k in present)
            k0 = present[0]
            bounds = []
            for b in range(int(cells[k0]["n_blocks"])):
                sb = cell_block(cells[k0], b)
                bounds.append(f"{BLOCK_LABELS[b]}:{int(sb['seeds'][0])}"
                              f"-{int(sb['seeds'][-1])}")
            tag = f"L={L},N_c={N_c}"
            res["block_inventory"][tag] = dict(R=Rs, full_blocks=nb,
                                               primary_R=BLOCK,
                                               cells_present=len(present))
            print(f"  {tag:<22}{str(Rs):>12}{nb:>13}{BLOCK:>11}   "
                  f"{'  '.join(bounds) if bounds else '(none)'}")

    # ---------------- A. the curves --------------------------------------
    hdr(f"A.  CMI(lambda) CURVES — PRIMARY, MATCHED R = {BLOCK} (block A)")
    res["curves"] = {}
    res["curves_full_R"] = {}
    quality = {}
    quality_full = {}
    for N_c in (NC_MAIN, NC_LOW):
        for L in LS:
            cs = curve(cells, L, N_c, block=PRIMARY_BLOCK)
            tag = f"L={L},N_c={N_c}"
            if cs is None:
                have = sum(1 for l in GRID if _cellkey(L, N_c, l) in cells)
                print(f"\n  [{tag}]  INCOMPLETE — {have}/13 grid points with a "
                      f"full R={BLOCK} block")
                continue
            qq = curve_quality(cs, rng)
            quality[(L, N_c)] = qq
            res["curves"][tag] = qq
            parent_R = [int(cells[_cellkey(L, N_c, l)]["R"]) for l in GRID]
            print(f"\n  [{tag}]   primary R = {BLOCK} at every lambda "
                  f"(drawn from cells of R = {sorted(set(parent_R))})")
            print("     lambda    R  fullR   mean CMI       SEM       VIF   "
                  "|  d      SEM(d)      r")
            for i, (lam, c) in enumerate(zip(GRID, cs)):
                tail = (f"| {qq['d'][i]:+8.5f} {qq['sem_d'][i]:8.5f} "
                        f"{qq['r'][i]:7.2f}") if i < 12 else "|"
                print(f"     {lam:6.4f} {c['R']:>4} {parent_R[i]:>6} "
                      f"{c['mean']:10.5f} {c['sem']:9.5f} {c['vif']:9.2f}   {tail}")
            print(f"     median r = {qq['r_median']:.2f}   "
                  f"increments with r>=2: {qq['n_r_ge_2']}/12")
            print(f"     roughness = mean (q/SEM q)^2 = {qq['roughness']:.3f}  "
                  f"bootstrap 95% CI [{qq['roughness_ci'][0]:.3f}, "
                  f"{qq['roughness_ci'][1]:.3f}]")
            print(f"     chi2/dof vs weighted quadratic YARDSTICK = "
                  f"{qq['quad_chi2_dof']:.3f}   (the historical dtau=12 corpus")
            print(f"       gives 0.60-1.38 by the same statistic; the quadratic is")
            print(f"       never plotted in place of the points)")

            # SECONDARY: the same curve at full R, for the mean only.
            fs = curve(cells, L, N_c, block=None)
            if fs is not None and any(c["R"] > BLOCK for c in fs):
                qf = curve_quality(fs, rng)
                quality_full[(L, N_c)] = qf
                res["curves_full_R"][tag] = qf
                print(f"     SECONDARY, full R = {sorted(set(parent_R))}: "
                      f"median r = {qf['r_median']:.2f}, "
                      f"roughness = {qf['roughness']:.3f}, "
                      f"chi2/dof = {qf['quad_chi2_dof']:.3f}")
                print( "       ^ higher-precision MEANS only. Its roughness and r are")
                print( "         NOT comparable across N_c classes with different R and")
                print( "         carry no 'cleaner curve' authority.")

    # ---------------- B. population diagnostics --------------------------
    hdr(f"B.  PER-CELL POPULATION DIAGNOSTICS  (M1) — PRIMARY, R = {BLOCK}")
    res["population_diagnostics"] = {}
    for L in LS:
        cs = curve(cells, L, NC_MAIN, block=PRIMARY_BLOCK)
        if cs is None:
            print(f"  L={L}: incomplete"); continue
        pd = population_diagnostics(cs, rng)
        res["population_diagnostics"][f"L={L}"] = pd
        print(f"\n  [L={L}, N_c={NC_MAIN}, block A]   split-half |m_A-m_B| vs "
              f"2.5*s_AB ; z_max vs z_R ; leave-one-out shift in SEM")
        for e in pd:
            flag = "".join(["" if e["split_half_ok"] else " SPLIT-HALF",
                            "" if e["outlier_ok"] else " OUTLIER",
                            "" if e["loo_ok"] else " LEAVE-ONE-OUT"])
            print(f"     {e['lam']:6.4f}  R={e['R']:>3}  "
                  f"dsh={e['split_half']:+8.5f}+-{e['split_half_sem']:.5f}  "
                  f"zmax={e['zmax']:.2f}/{e['z_thr']:.2f}  "
                  f"loo={e['loo_shift_sem']:.2f}  "
                  f"{'ok' if not flag else '** FAIL:' + flag}")

    # ---------------- C. crossings ---------------------------------------
    hdr(f"C.  CROSSING ANALYSIS  (brief section 11) — PRIMARY, MATCHED R = {BLOCK}")
    print("  LOCATOR QUALITY ONLY. CMI is the locator. Nothing here is")
    print("  lambda_c(zeta) and nothing here may be promoted into a physical")
    print("  phase boundary, however stable.")
    print(f"  Both N_c classes are compared at the SAME R = {BLOCK}, so a")
    print("  difference in crossing count or uniqueness cannot be an artefact")
    print("  of one class having more independent populations than the other.")
    res["crossings"] = {}
    for N_c in (NC_MAIN, NC_LOW):
        for (L1, L2) in PAIRS:
            c1 = curve(cells, L1, N_c, block=PRIMARY_BLOCK)
            c2 = curve(cells, L2, N_c, block=PRIMARY_BLOCK)
            lab = f"L{L1}-L{L2}@Nc{N_c}"
            if c1 is None or c2 is None:
                print(f"\n  [{lab}] incomplete"); continue
            x = crossing_analysis(c1, c2, rng, lab)
            res["crossings"][lab] = x
            print(f"\n  [{lab}]  D = I_{L2} - I_{L1}")
            print(f"     raw sign changes      {x['n_raw']}")
            print(f"     resolved (both |D|>=2 SEM)  {x['n_resolved']}")
            print(f"     crossing lambda(s)    "
                  f"{[round(v, 5) for v in x['crossings']]}")
            print(f"     bootstrap 95% CI      "
                  f"[{x['boot_ci'][0]:.5f}, {x['boot_ci'][1]:.5f}]  "
                  f"width {x['boot_width']:.5f}")
            print(f"     bootstrap crossing-count histogram  "
                  f"{x['boot_count_hist']}   (never discarded)")
            print(f"     unique                {x['unique']}  "
                  f"(fraction with exactly one = {x['frac_exactly_one']:.3f})")
            print(f"     endpoint-induced      {x['endpoint_induced']}")
            print(f"     stable to bootstrap   {x['stable_bootstrap']}  "
                  f"(needs width <= {2 * DLAM})")
            print(f"     stable to deleting one lambda  "
                  f"{x['stable_point_deletion']}")

    # ---------------- D. N_c comparison ----------------------------------
    hdr("D.  HIGH-N_c versus LOW-N_c  (brief section 12)")
    res["nc_comparison"] = []
    n_common = sum(1 for L in LS for l in GRID
                   if _cellkey(L, NC_MAIN, l) in cells
                   and _cellkey(L, NC_LOW, l) in cells)
    print(f"  exact common (N_c=1024, N_c=128) cells: {n_common} of 39")
    print("  exact common cells against the HISTORICAL corpus: 0 of 39 —")
    print("  the corpus is dtau_mult = 12.0 throughout and this campaign is the")
    print("  certified 6.0. Nothing below uses it quantitatively.")
    res["nc_comparison_full_R"] = []
    for L in LS:
        hi = curve(cells, L, NC_MAIN, block=PRIMARY_BLOCK)
        lo = curve(cells, L, NC_LOW, block=PRIMARY_BLOCK)
        if hi is None or lo is None:
            print(f"\n  [L={L}] incomplete"); continue
        e = nc_comparison(hi, lo, L)
        res["nc_comparison"].append(e)
        print(f"\n  [L={L}]  Delta(lambda) = I_1024 - I_128")
        print(f"     PRIMARY, matched R = {BLOCK} on both sides:")
        print(f"       weighted mean shift   {e['mean_shift']:+.5f} "
              f"+- {e['mean_shift_sem']:.5f}")
        print(f"       chi2/dof constant     {e['chi2_const_dof']:.3f}")
        print(f"       chi2/dof + linear     {e['chi2_linear_dof']:.3f}")
        print(f"       chi2/dof + quadratic  {e['chi2_quad_dof']:.3f}")
        print(f"       -> {e['classification']}")
        # SECONDARY: the highest-precision mean displacement available. The
        # brief asks for BOTH. Unequal R is stated in the line itself.
        hf = curve(cells, L, NC_MAIN, block=None)
        lf = curve(cells, L, NC_LOW, block=None)
        if hf is not None and lf is not None:
            ef = nc_comparison(hf, lf, L)
            ef["unequal_R"] = True
            ef["R_high"] = sorted({int(c["R"]) for c in hf})
            ef["R_low"] = sorted({int(c["R"]) for c in lf})
            res["nc_comparison_full_R"].append(ef)
            print(f"     SECONDARY, highest precision available "
                  f"(R_1024 = {ef['R_high']}, R_128 = {ef['R_low']}, UNEQUAL R):")
            print(f"       weighted mean shift   {ef['mean_shift']:+.5f} "
                  f"+- {ef['mean_shift_sem']:.5f}")
            print( "       ^ mean DISPLACEMENT only. Its chi2/dof shape "
                   "classification is")
            print( "         reported in the JSON but carries no authority over "
                   "the primary.")
    for L in LS:
        q = quality.get((L, NC_LOW))
        if q:
            v = q["quad_chi2_dof"]
            print(f"\n  jaggedness question, L={L}, N_c=128 at matched R={BLOCK}: "
                  f"chi2/dof vs quadratic = {v:.3f}")
            print("     -> " + ("the apparent jaggedness IS explained by the "
                                "independent-population uncertainty"
                                if 0.5 <= v <= 1.5 else
                                "NOT explained by the independent-population "
                                "uncertainty alone"))
            qh = quality.get((L, NC_MAIN))
            if qh:
                print(f"     matched-R comparison: chi2/dof is "
                      f"{qh['quad_chi2_dof']:.3f} at N_c=1024 vs {v:.3f} at "
                      f"N_c=128,")
                print(f"       and roughness {qh['roughness']:.3f} vs "
                      f"{q['roughness']:.3f}. Both sides R = {BLOCK}, so any")
                print( "       difference is attributable to N_c and not to R.")

    # ---------------- E. Delta_N shape check -----------------------------
    hdr(f"E.  N_c = 2048 SHAPE CHECK AT L = 64  (brief section 6, M5)"
        f" — PRIMARY R = {BLOCK}")
    ks1 = [_cellkey(64, NC_MAIN, l) for l in CENTRE3]
    ks2 = [_cellkey(64, NC_HIGH, l) for l in CENTRE3]
    if all(k in cells for k in ks1 + ks2):
        # PRIMARY: block A on both sides. The N_c=2048 arm is R=24 by design,
        # so this is the matched comparison; the N_c=1024 side is cut down from
        # R=96 to its block A rather than being allowed to be four times as
        # precise as the thing it is subtracted from.
        b1 = [cell_block(cells[k], PRIMARY_BLOCK) for k in ks1]
        b2 = [cell_block(cells[k], PRIMARY_BLOCK) for k in ks2]
        if any(b is None for b in b1 + b2):
            print(f"     NOT EVALUATED — a centre cell has fewer than {BLOCK} "
                  f"populations.")
            b1 = b2 = None
    else:
        b1 = b2 = None
    if b1 is not None:
        cells1 = {k: b for k, b in zip(ks1, b1)}
        cells2 = {k: b for k, b in zip(ks2, b2)}
        m1 = np.array([cells1[k]["mean"] for k in ks1])
        m2 = np.array([cells2[k]["mean"] for k in ks2])
        s1 = np.array([cells1[k]["sem"] for k in ks1])
        s2 = np.array([cells2[k]["sem"] for k in ks2])
        D = m2 - m1
        sD = np.sqrt(s1 ** 2 + s2 ** 2)
        W = 1 / sD ** 2
        cbar = float(np.sum(W * D) / np.sum(W))
        chi2 = float(np.sum(W * (D - cbar) ** 2))
        # The bootstrap NULL must be a world in which Delta_N really is constant.
        # Resampling the observed populations directly would carry the observed
        # lambda-dependence into the "null", which inflates p and systematically
        # hides exactly the effect M5 exists to detect. So each N_c=2048 cell's
        # populations are shifted by -(D_j - cbar) before resampling: the null
        # becomes true by construction while the empirical spread is preserved.
        null2 = []
        for k, dj in zip(ks2, D):
            c = dict(cells2[k])
            c["pops"] = cells2[k]["pops"] - (dj - cbar)
            null2.append(c)
        rng_null = np.random.default_rng(SEED)
        b1n = boot_curves([cells1[k] for k in ks1], rng_null, B_BOOT)
        b2n = boot_curves(null2, rng_null, B_BOOT)
        bDn = b2n - b1n
        cbar_n = (bDn * W).sum(1, keepdims=True) / W.sum()
        bchi = np.sum(W * (bDn - cbar_n) ** 2, axis=1)
        p = float(np.mean(bchi >= chi2))

        # The slope BOUND, by contrast, is a confidence statement about the
        # observed Delta_N, so it is bootstrapped around the observed data.
        x = np.array(CENTRE3) - float(np.mean(CENTRE3))
        slope = float(np.sum(W * x * (D - cbar)) / np.sum(W * x ** 2))
        bb1 = boot_curves([cells1[k] for k in ks1], rng, B_BOOT)
        bb2 = boot_curves([cells2[k] for k in ks2], rng, B_BOOT)
        bD = bb2 - bb1
        bslope = ((bD - (bD * W).sum(1, keepdims=True) / W.sum()) * W * x).sum(1) \
            / np.sum(W * x ** 2)
        sl_hi = float(np.percentile(np.abs(bslope), 95)) * DLAM
        med_d = float(np.median(np.abs(quality[(64, NC_MAIN)]["d"]))) \
            if (64, NC_MAIN) in quality else float("nan")
        tau_shape = 0.2 * med_d
        verdict = ("A: approximately a common shift across lambda"
                   if p > 0.32 and sl_hi <= tau_shape else
                   "B: appreciably lambda-dependent" if p < 0.05 else
                   "C: unresolved")
        res["delta_N"] = dict(lams=CENTRE3, delta=D.tolist(), sem=sD.tolist(),
                              common_shift=cbar, chi2_const=chi2, boot_p=p,
                              slope=slope, slope_95_times_dlam=sl_hi,
                              tau_shape=tau_shape, verdict=verdict,
                              matched_R=BLOCK,
                              R_1024=int(b1[0]["R"]), R_2048=int(b2[0]["R"]))
        print(f"     PRIMARY, matched R = {BLOCK} on both sides "
              f"(N_c=1024 cut from R={int(cells[ks1[0]]['R'])} to its block A):")
        for lam, d, s in zip(CENTRE3, D, sD):
            print(f"       lambda={lam:6.4f}   Delta_N = {d:+.5f} +- {s:.5f}")
        print(f"       weighted common shift {cbar:+.5f}")
        print(f"       chi2 vs constant (2 dof) = {chi2:.3f}, bootstrap p = {p:.4f}")
        print(f"       95% bound on |slope|*dlambda = {sl_hi:.5f}  "
              f"vs tau_shape = {tau_shape:.5f}")
        print(f"       -> {verdict}")

        # SECONDARY: the highest-precision mean displacement. The brief asks
        # for BOTH. Only the MEAN is reported here -- the shape verdict stays
        # with the matched-R analysis above.
        mf1 = np.array([cells[k]["mean"] for k in ks1])
        sf1 = np.array([cells[k]["sem"] for k in ks1])
        Df = np.array([cells2[k]["mean"] for k in ks2]) - mf1
        sDf = np.sqrt(sf1 ** 2 + np.array([cells2[k]["sem"] for k in ks2]) ** 2)
        Wf = 1 / sDf ** 2
        cbar_f = float(np.sum(Wf * Df) / np.sum(Wf))
        res["delta_N_full_R"] = dict(
            lams=CENTRE3, delta=Df.tolist(), sem=sDf.tolist(),
            common_shift=cbar_f,
            R_1024=[int(cells[k]["R"]) for k in ks1],
            R_2048=[int(cells2[k]["R"]) for k in ks2], unequal_R=True)
        print(f"     SECONDARY, highest precision available "
              f"(R_1024 = {int(cells[ks1[0]]['R'])}, "
              f"R_2048 = {int(b2[0]['R'])}, UNEQUAL R):")
        for lam, d, s in zip(CENTRE3, Df, sDf):
            print(f"       lambda={lam:6.4f}   Delta_N = {d:+.5f} +- {s:.5f}")
        print(f"       weighted common shift {cbar_f:+.5f}")
        print( "       ^ mean finite-N_c DISPLACEMENT only. The A/B/C verdict "
               "stays with")
        print( "         the matched-R analysis. More R does not remove "
               "finite-N_c bias.")
        print("     No 1/N_c law is fitted. Delta_N is not extrapolated to "
              "N_c = infinity.")
    elif not all(k in cells for k in ks1 + ks2):
        print("     NOT EVALUATED — the N_c=2048 arm and/or the N_c=1024 centre "
              "is incomplete.")

    # ---------------- E2. reused ARM-B block sensitivity ------------------
    hdr("E2.  REUSED ARM-B BLOCK SENSITIVITY  (amendment section 3)")
    print("  The three reused cells hold R=96 and cut into four disjoint R=24")
    print("  blocks in seed order. Block A is PRIMARY because it is")
    print("  deterministic and observable-blind. B/C/D are SENSITIVITY checks —")
    print("  they are NOT an opportunity to choose the nicest-looking subset,")
    print("  and no verdict anywhere in this file is taken from them.")
    res["armB_block_sensitivity"] = {}
    ksC = [_cellkey(64, NC_MAIN, l) for l in CENTRE3]
    if all(k in cells for k in ksC):
        nb = min(int(cells[k]["n_blocks"]) for k in ksC)
        print(f"\n  {'block':<7}{'seeds':>22}" +
              "".join(f"{'  lam=' + format(l, '.4f'):>18}" for l in CENTRE3))
        rows_by_block = {}
        for b in range(nb):
            subs = [cell_block(cells[k], b) for k in ksC]
            rows_by_block[BLOCK_LABELS[b]] = subs
            seeds = f"{int(subs[0]['seeds'][0])}-{int(subs[0]['seeds'][-1])}"
            cellstr = "".join(f"  {s['mean']:8.5f}+-{s['sem']:.5f}" for s in subs)
            tag = BLOCK_LABELS[b] + (" *" if b == PRIMARY_BLOCK else "  ")
            print(f"  {tag:<7}{seeds:>22}{cellstr}")
        full = [cells[k] for k in ksC]
        print(f"  {'FULL':<7}{'R=96 (secondary)':>22}" +
              "".join(f"  {c['mean']:8.5f}+-{c['sem']:.5f}" for c in full))
        print("  * = primary")

        # adjacent increments across the three reused points, per block
        print(f"\n  adjacent increments d over the reused triple, per block")
        print(f"  {'block':<7}{'d(0.2932->0.3032)':>22}{'d(0.3032->0.3132)':>22}"
              f"{'q':>12}")
        for lab, subs in rows_by_block.items():
            m = np.array([s["mean"] for s in subs])
            s = np.array([s["sem"] for s in subs])
            d1, d2 = m[1] - m[0], m[2] - m[1]
            sd1 = math.hypot(s[0], s[1]); sd2 = math.hypot(s[1], s[2])
            q = m[2] - 2 * m[1] + m[0]
            print(f"  {lab:<7}{d1:+11.5f}+-{sd1:.5f}{d2:+11.5f}+-{sd2:.5f}"
                  f"{q:+12.5f}")
            res["armB_block_sensitivity"][lab] = dict(
                seeds=[int(subs[0]["seeds"][0]), int(subs[0]["seeds"][-1])],
                means=m.tolist(), sems=s.tolist(),
                d=[float(d1), float(d2)], sem_d=[float(sd1), float(sd2)],
                q=float(q))

        # do the block-dependent quantities move the whole-curve conclusions?
        if curve(cells, 64, NC_MAIN, block=PRIMARY_BLOCK) is not None:
            print(f"\n  effect on the whole L=64 curve of swapping block A for "
                  f"B/C/D at the three reused points:")
            print(f"  {'block':<7}{'median r':>10}{'roughness':>12}"
                  f"{'chi2/dof':>11}{'n_raw(32-64)':>14}{'n_raw(48-64)':>14}")
            res["armB_block_curve_effect"] = {}
            for b in range(nb):
                cs = []
                ok = True
                for l in GRID:
                    k = _cellkey(64, NC_MAIN, l)
                    use = b if l in CENTRE3 else PRIMARY_BLOCK
                    sb = cell_block(cells[k], use)
                    if sb is None:
                        ok = False; break
                    cs.append(sb)
                if not ok:
                    continue
                qq = curve_quality(cs, np.random.default_rng(SEED))
                ent = dict(r_median=qq["r_median"], roughness=qq["roughness"],
                           quad_chi2_dof=qq["quad_chi2_dof"])
                nraw = {}
                for (L1, L2) in ((32, 64), (48, 64)):
                    c1 = curve(cells, L1, NC_MAIN, block=PRIMARY_BLOCK)
                    if c1 is None:
                        nraw[f"{L1}-{L2}"] = None; continue
                    mA = np.array([c["mean"] for c in c1])
                    Dv = np.array([c["mean"] for c in cs]) - mA
                    nraw[f"{L1}-{L2}"] = len(_crossings_of(Dv.tolist(), GRID))
                ent["n_raw"] = nraw
                res["armB_block_curve_effect"][BLOCK_LABELS[b]] = ent
                f32 = nraw.get("32-64"); f48 = nraw.get("48-64")
                print(f"  {BLOCK_LABELS[b] + (' *' if b == PRIMARY_BLOCK else '  '):<7}"
                      f"{qq['r_median']:10.2f}{qq['roughness']:12.3f}"
                      f"{qq['quad_chi2_dof']:11.3f}"
                      f"{str(f32):>14}{str(f48):>14}")
            print("  A material dependence on the block would mean the three")
            print("  reused points, not the campaign, are driving the result.")
    else:
        print("  NOT EVALUATED — the reused ARM-B cells are not loaded.")

    # ---------------- E3. historical-precision (R = 12) checks ------------
    hdr(f"E3.  HISTORICAL-PRECISION CHECKS  (R = {HIST_PRECISION_R}) — SECONDARY")
    print(f"  These mimic the historical corpus's own R = {HIST_PRECISION_R} and")
    print("  exist ONLY to ask whether R=12 is what made the old scan look")
    print(f"  jagged. They cannot replace the primary R = {BLOCK} comparison and")
    print("  no M-criterion reads them.")
    res["hist_precision"] = {}
    for N_c in (NC_MAIN, NC_LOW):
        for L in LS:
            ks = [_cellkey(L, N_c, l) for l in GRID]
            if any(k not in cells for k in ks):
                continue
            nb = min(int(cells[k]["R"]) // HIST_PRECISION_R for k in ks)
            if nb < 1:
                continue
            vals = []
            for b in range(nb):
                cs = [cell_block(cells[k], b, size=HIST_PRECISION_R) for k in ks]
                if any(c is None for c in cs):
                    continue
                qq = curve_quality(cs, np.random.default_rng(SEED + b))
                vals.append((qq["roughness"], qq["quad_chi2_dof"]))
            if not vals:
                continue
            ro = [v[0] for v in vals]; ch = [v[1] for v in vals]
            res["hist_precision"][f"L={L},N_c={N_c}"] = dict(
                n_subsets=len(vals), roughness=ro, quad_chi2_dof=ch)
            print(f"  L={L}, N_c={N_c}: {len(vals)} disjoint R={HIST_PRECISION_R} "
                  f"subsets, roughness {min(ro):.3f}-{max(ro):.3f}, "
                  f"chi2/dof {min(ch):.3f}-{max(ch):.3f}")
    if not res["hist_precision"]:
        print("  none available yet.")

    # ---------------- F. success criteria --------------------------------
    hdr("F.  PRE-REGISTERED SUCCESS CRITERIA  M1-M7")
    print(f"  All curve-quality and crossing criteria below are evaluated on the")
    print(f"  PRIMARY matched R = {BLOCK} analysis. M3 in particular compares")
    print(f"  N_c=128 and N_c=1024 at identical R, so it cannot reward one class")
    print("  merely for having more independent populations.")
    res["criteria"] = {}

    def emit(k, v, why):
        res["criteria"][k] = dict(verdict=v, reason=why)
        print(f"  {k}  -> {v}")
        print(f"        {why}")

    # M1
    pdg = res["population_diagnostics"]
    if len(pdg) < 3:
        emit("M1", "NOT EVALUATED", "not all three N_c=1024 curves are complete")
    else:
        fails_sh = {L: sum(1 for e in v if not e["split_half_ok"])
                    for L, v in pdg.items()}
        fails_loo = sum(1 for v in pdg.values() for e in v if not e["loo_ok"])
        fails_out = sum(1 for v in pdg.values() for e in v if not e["outlier_ok"])
        if max(fails_sh.values()) == 0 and fails_loo == 0 and fails_out == 0:
            emit("M1", "SUPPORTED", "split-half, outlier and leave-one-out all "
                                    "pass at every (L, lambda)")
        elif max(fails_sh.values()) >= 2 or fails_loo > 0:
            emit("M1", "KILLED", f"split-half failures per L {fails_sh}, "
                                 f"leave-one-out failures {fails_loo}")
        else:
            emit("M1", "INCONCLUSIVE", f"isolated failures: split-half "
                                       f"{fails_sh}, outlier {fails_out}")
    # M2
    qs = {L: quality.get((L, NC_MAIN)) for L in LS}
    if any(v is None for v in qs.values()):
        emit("M2", "NOT EVALUATED", "not all three N_c=1024 curves are complete")
    else:
        med = {L: qs[L]["r_median"] for L in LS}
        cnt = {L: qs[L]["n_r_ge_2"] for L in LS}
        if all(2 <= med[L] <= 20 for L in LS) and all(cnt[L] >= 9 for L in LS):
            emit("M2", "SUPPORTED", f"median r {med}, increments with r>=2 {cnt}")
        elif any(med[L] < 2 for L in LS) or all(med[L] > 20 for L in LS):
            emit("M2", "KILLED", f"median r {med}")
        else:
            emit("M2", "INCONCLUSIVE", f"median r {med}, r>=2 counts {cnt}")
    # M3
    hi = {p: res["crossings"].get(f"L{p[0]}-L{p[1]}@Nc{NC_MAIN}") for p in PAIRS}
    lo = {p: res["crossings"].get(f"L{p[0]}-L{p[1]}@Nc{NC_LOW}") for p in PAIRS}
    if any(v is None for v in hi.values()):
        emit("M3", "NOT EVALUATED", "the N_c=1024 crossing set is incomplete")
    elif any(v is None for v in lo.values()):
        emit("M3", "INCONCLUSIVE",
             "the matched N_c=128 companion arm was not run, so M3 has no "
             "matched comparator. Comparing against the dtau_mult=12 historical "
             "corpus instead would not be a matched comparison and is refused.")
    else:
        sh = sum(hi[p]["n_raw"] for p in PAIRS)
        sl = sum(lo[p]["n_raw"] for p in PAIRS)
        per_ok = all(hi[p]["n_raw"] <= lo[p]["n_raw"] for p in PAIRS)
        got_one = any(hi[p]["unique"] and not hi[p]["endpoint_induced"]
                      and hi[p]["stable_bootstrap"] for p in PAIRS)
        m = f"at MATCHED R = {BLOCK} on both sides"
        if per_ok and sh <= 0.5 * sl and got_one:
            emit("M3", "SUPPORTED", f"raw sign changes {sh} at N_c=1024 vs {sl} "
                                    f"at N_c=128 {m}, and a unique stable "
                                    f"non-endpoint crossing exists")
        elif sh >= sl:
            emit("M3", "KILLED", f"raw sign changes {sh} at N_c=1024 vs {sl} "
                                 f"at N_c=128 {m}")
        else:
            emit("M3", "INCONCLUSIVE", f"raw sign changes {sh} vs {sl} {m}; "
                                       f"per-pair {per_ok}, unique stable "
                                       f"crossing {got_one}")
    # M4
    if len(pdg) < 3:
        emit("M4", "NOT EVALUATED", "not all three N_c=1024 curves are complete")
    else:
        worst = max(abs(e["split_half"]) / e["split_half_sem"]
                    for v in pdg.values() for e in v if e["split_half_sem"] > 0)
        emit("M4", "SUPPORTED" if worst <= 3.0 else "KILLED",
             f"max |I_A - I_B| / s_AB over all (L, lambda) = {worst:.2f} "
             f"(threshold 3.0); the crossing half-comparison is reported in "
             f"section C of the split-half report")
    # M5. "Data has not arrived yet" is NOT EVALUATED; "data arrived and cannot
    # be used" is KILLED. Conflating them would let an unsubmitted arm read as
    # a scientific verdict.
    dn = res.get("delta_N")
    n2048 = sum(1 for l in CENTRE3 if _cellkey(64, NC_HIGH, l) in cells)
    if not dn and n2048 == 0:
        emit("M5", "NOT EVALUATED", "the N_c=2048 arm has returned no results")
    elif not dn:
        emit("M5", "KILLED", f"the N_c=2048 arm returned {n2048}/3 usable cells; "
                             f"the shape check cannot be evaluated")
    elif dn["verdict"].startswith("C"):
        emit("M5", "INCONCLUSIVE", dn["verdict"])
    else:
        emit("M5", "SUPPORTED", dn["verdict"])
    # M6
    ok6 = (not AUDIT["smoothing_applied"] and AUDIT["value_based_exclusions"] == 0
           and AUDIT["lambda_points_removed"] == 0)
    emit("M6", "SUPPORTED" if ok6 else "KILLED",
         f"smoothing_applied={AUDIT['smoothing_applied']}, "
         f"value_based_exclusions={AUDIT['value_based_exclusions']}, "
         f"lambda_points_removed={AUDIT['lambda_points_removed']}")
    # M7 -- needs the measured rates, which arrive with the results
    walls = {}
    for L in LS:
        # timing, not a statistic: use every population that ran, not block A
        cs = curve(cells, L, NC_MAIN, block=None)
        if cs is None:
            continue
        # ms per clone-window from the returned wall times
        ns = [_nsteps(L, L, l) for l in GRID]
        w = [float(np.median(c["wall"])) for c in cs]
        walls[L] = float(np.median([wi * 1000.0 / (NC_MAIN * n)
                                    for wi, n in zip(w, ns)]))
    if len(walls) < 3:
        emit("M7", "NOT EVALUATED", "measured rates for all three L are needed")
    else:
        xs = np.log(np.array(sorted(walls)))
        ys = np.log(np.array([walls[L] for L in sorted(walls)]))
        p_exp, lnA = np.polyfit(xs, ys, 1)
        pred = {L: math.exp(lnA + p_exp * math.log(L)) for L in (96, 128)}
        resid = max(abs(math.exp(lnA + p_exp * math.log(L)) / walls[L] - 1)
                    for L in walls)
        core96 = sum(pred[96] * 1e-3 * NC_MAIN * _nsteps(96, 96, l) * 24
                     for l in GRID) / 3600
        el96 = max(core96 / 64 * 1.15,
                   pred[96] * 1e-3 * NC_MAIN * _nsteps(96, 96, GRID[-1]) / 3600)
        res["M7_projection"] = dict(measured_rates_ms=walls, exponent=float(p_exp),
                                    max_anchor_residual=float(resid),
                                    L96_core_h=core96, L96_elapsed_h_at_64=el96)
        print("  M7 projection, from the rates this campaign itself measured:")
        print(f"        measured rates ms/clone-window: "
              f"{ {k: round(v, 3) for k, v in walls.items()} }")
        print(f"        re-derived L exponent {p_exp:.3f}, max anchor residual "
              f"{100 * resid:.1f} %")
        print(f"        projected L=96 13-point N_c=1024 R=24 scan: "
              f"{core96:.0f} core-h, {el96:.1f} h elapsed at %64")
        if resid <= 0.15 and el96 <= 24:
            emit("M7", "SUPPORTED", f"model reproduces its anchors to "
                                    f"{100 * resid:.1f} % and L=96 needs "
                                    f"{el96:.1f} h elapsed at %64")
        elif el96 > 72:
            emit("M7", "KILLED", f"L=96 needs {el96:.1f} h elapsed at %64")
        else:
            emit("M7", "INCONCLUSIVE", f"anchor residual {100 * resid:.1f} %, "
                                       f"L=96 elapsed {el96:.1f} h")

    # ---------------- figures --------------------------------------------
    hdr("G.  FIGURES  (validation figures, NOT manuscript figures)")
    for p in figures(cells, res):
        print(f"  wrote {os.path.relpath(p, TASK)}")

    hdr("WHAT THIS ANALYSIS MAY NOT SAY")
    print("  * No crossing above is lambda_c(zeta = 0.35) or a finite-size")
    print("    estimate of it. L = 32, 48, 64 are BELOW the programme's own")
    print("    corpus floor of 64. The output is locator quality.")
    print("  * No global phase-boundary law, no finite-zeta exponent.")
    print("  * No 1/N_c bias law; Delta_N is a measured difference at three")
    print("    lambdas and nothing more.")
    print("  * The dtau_mult = 12 historical corpus is descriptive only and is")
    print("    never pooled, averaged or interpolated into any number above.")

    out = os.path.join(TASK, "MOCK_PRODUCTION_RESULTS.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("\n" + "=" * 78)
    print(f"  wrote {out}")
    print("=" * 78)
    return 0


def _nsteps(L, T, lam, dtau_mult=DTAU_MULT):
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


if __name__ == "__main__":
    sys.exit(main())
