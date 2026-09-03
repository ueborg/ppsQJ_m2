#!/usr/bin/env python3
"""Execute ONE manifest row. Called once per array task.

Runs the CERTIFIED production path (pps_qj) through the bundled instrumented
wrapper that TASK-2026-08-30-SMCSTAT validated bitwise against production and
TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION shipped byte-identical. Writes one JSON
row. Idempotent: a completed row is never recomputed.

WHAT IS DIFFERENT FROM THE PREDECESSOR'S run_cell.py, AND WHAT IS NOT
---------------------------------------------------------------------
NOT different: the sampler, its arguments, the RNG seeding, the discretisation,
the resampling scheme, the observable, or the order in which anything is called.
`support/instrumented.py` is byte-identical (sha256 checked below, hard failure
on mismatch) and the call is the same single line with the same keywords. A row
produced here is EXACT-COMPATIBLE with a row produced by the predecessor at the
same (L, T, zeta, lam, N_c, dtau_mult, resample_scheme, seed) -- and
../VALIDATION.md records the direct bit-level reproduction that demonstrates it
rather than asserting it.

Different: WHAT IS WRITTEN DOWN. TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING
found that the one quantity that could still explain the L-growth of finite-N_c
drift -- the across-clone spread of the ACCUMULATED log weight -- is recorded in
0 % of production-geometry runs, and that `git_commit` is absent from 100 % of
the 3784-record corpus. Both are fixed here, at zero cost to the simulation:

  * `final_weights` is persisted. It is the normalised cumulative weight vector,
    so log(final_weights) recovers the accumulated log weight up to an additive
    constant and Var(log w_carry) at t=T is EXACT, not a proxy.
    `logw_carry_var_final` is that variance, computed here.
  * the per-window diagnostic histories the sampler already computes and the
    predecessor threw away -- ess, ess_cum, logw_var, w_max, dLambda_mean,
    dLambda_var, n_jumps_mean, n_distinct_anc, gess, max_family_frac,
    resampled -- are persisted in full. These are O(n_steps) arrays, not
    O(n_steps x N_c): the largest file this campaign writes is about 230 kB.
  * `delta_tau`, `n_resampling_events` and `git_commit` are persisted.

Adding output fields cannot change a trajectory. Every field the predecessor
wrote is still written, under the same key, with the same definition, so every
existing analysis script reads these files unchanged.

This file contains no scheduler call and cannot submit.
"""
import os, sys, json, csv, time, hashlib, subprocess
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUPPORT = os.path.abspath(os.path.join(HERE, os.pardir, "support"))

# Repository root: five levels up from <repo>/research/tasks/active/<TASK>/<arm>.
# PPSQJ_REPO overrides it for an unusual layout, but is not required.
REPO = os.environ.get("PPSQJ_REPO") or os.path.abspath(
    os.path.join(HERE, *([os.pardir] * 5)))

_need = {
    "bundled instrumented.py": os.path.join(SUPPORT, "instrumented.py"),
    "bundle manifest":         os.path.join(SUPPORT, "BUNDLE_MANIFEST.json"),
    "pps_qj package":          os.path.join(REPO, "pps_qj", "__init__.py"),
}
_missing = {k: v for k, v in _need.items() if not os.path.isfile(v)}
if _missing:
    sys.stderr.write("run_cell.py cannot start. Missing required runtime files:\n")
    for _k, _v in _missing.items():
        sys.stderr.write("    %-24s %s\n" % (_k, _v))
    sys.stderr.write("  REPO resolved to: %s\n" % REPO)
    sys.stderr.write("  If the layout differs, export PPSQJ_REPO to the repo root.\n")
    sys.exit(2)

# The bundle must be the EXACT file that produced the reused populations. A
# silent substitution here would change the sampler without changing any
# manifest row, and the whole reuse ledger would become fiction.
_man = json.load(open(_need["bundle manifest"]))
for _f in _man["files"]:
    _p = os.path.join(SUPPORT, os.path.basename(_f["bundled_as"]))
    _h = hashlib.sha256(open(_p, "rb").read()).hexdigest()
    if _h != _f["sha256_bundled"]:
        sys.exit("INTEGRITY FAILURE: %s\n  expected sha256 %s\n  found    sha256 %s\n"
                 "  The bundled instrumentation is not the file that produced the "
                 "reused populations. Refusing to run." % (_p, _f["sha256_bundled"], _h))

sys.path.insert(0, REPO)      # the TRACKED pps_qj package
sys.path.insert(0, SUPPORT)   # the TRACKED bundled instrumentation
import instrumented as I
print("[env] python       %s" % sys.executable)
print("[env] instrumented %s" % os.path.abspath(I.__file__))
print("[env] pps_qj       %s" % __import__("pps_qj").__file__)


def git_commit(repo):
    """The code version this row was produced by. Absent from 100 % of the
    existing corpus, which is why no historical run can be tied to its code.
    Never fatal: a tarball unpacked on the cluster has no .git."""
    try:
        h = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        if h.returncode != 0:
            return "unavailable"
        d = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=20)
        return h.stdout.strip() + ("-dirty" if d.stdout.strip() else "")
    except Exception:
        return "unavailable"


def arr(x, nd=6):
    a = np.asarray(x)
    if a.dtype == bool:
        return [bool(v) for v in a]
    if np.issubdtype(a.dtype, np.integer):
        return [int(v) for v in a]
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


idx = int(sys.argv[1])
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
cfg = rows[idx]
out_path = os.path.join(outdir, f"{cfg['arm']}_{idx:05d}.json")
if os.path.exists(out_path):
    print(f"[cached] {out_path}")
    sys.exit(0)

kw = dict(L=int(cfg["L"]), T=float(cfg["T"]), N_c=int(cfg["N_c"]),
          zeta=float(cfg["zeta"]), lam=float(cfg["lam"]),
          dtau_mult=float(cfg["dtau_mult"]), seed=int(cfg["seed"]),
          resample_scheme=cfg["resample_scheme"])
t0 = time.time()
r = I.run_instrumented(**kw, record_anc=True)
f = np.asarray(r.obs["CMI"], float)
w = np.asarray(r.final_weights, float)
ok = np.isfinite(f)

row = dict(cfg)
# ---- every field the predecessor wrote, unchanged in key and definition -----
row.update(status="ok", wall_s=round(time.time() - t0, 2), n_steps=r.n_steps,
           cmi_weighted_mean=float(np.sum(w[ok] * f[ok]) / np.sum(w[ok])),
           cmi_unweighted_mean=float(np.mean(f[ok])),
           cmi_within_var=float(np.var(f[ok], ddof=1)),
           n_nonfinite=int((~ok).sum()),
           n_distinct_anc_final=int(np.asarray(r.n_distinct_anc)[-1]),
           gess_final=float(np.asarray(r.gess)[-1]),
           ess_cum_final=float(np.asarray(r.ess_cum)[-1]),
           ess_frac_mean=float(np.mean(np.asarray(r.ess)) / int(cfg["N_c"])),
           brentq_fallbacks=int(r.brentq_fallbacks),
           per_clone_CMI=[None if not np.isfinite(x) else round(float(x), 9)
                          for x in f])
# ---- new, output-only, requested by TASK-2026-09-02-FINITE-NC-THEORY --------
lw = np.log(np.clip(w, 1e-300, None))
row.update(
    delta_tau=float(r.delta_tau),
    K=int(r.n_steps),                       # windows; named K in the theory task
    n_resampling_events=int(r.n_resampling_events),
    resample_mode=r.resample_mode,
    git_commit=git_commit(REPO),
    sampler_sha256=_man["files"][0]["sha256_bundled"],
    # EXACT accumulated-log-weight spread at t = T. Recorded in 0 % of the
    # existing production-geometry corpus; the one surviving candidate factor
    # for the L-growth of finite-N_c drift could not be tested without it.
    logw_carry_var_final=float(np.var(lw)),
    final_weights=arr(w, 12),
    # per-window histories: O(n_steps), not O(n_steps * N_c)
    hist_ess=arr(r.ess, 4), hist_ess_cum=arr(r.ess_cum, 4),
    hist_logw_var=arr(r.logw_var), hist_w_max=arr(r.w_max),
    hist_dLambda_mean=arr(r.dLambda_mean), hist_dLambda_var=arr(r.dLambda_var),
    hist_n_jumps_mean=arr(r.n_jumps_mean), hist_n_distinct_anc=arr(r.n_distinct_anc),
    hist_gess=arr(r.gess, 4), hist_max_family_frac=arr(r.max_family_frac),
    hist_resampled=arr(r.resampled))
json.dump(row, open(out_path, "w"))
print(f"[ok] idx={idx} L={cfg['L']} N_c={cfg['N_c']} lam={cfg['lam']} "
      f"dtau_mult={cfg['dtau_mult']} K={r.n_steps} wall={row['wall_s']}s -> {out_path}")
