#!/usr/bin/env python3
"""Execute ONE manifest row. Called once per array task.

Runs the CERTIFIED production path (pps_qj) through the parent's instrumented
wrapper, which TASK-2026-08-30-SMCSTAT validated bitwise against production.
Writes one JSON row. Idempotent: a completed row is never recomputed.
"""
import os, sys, json, csv, time, hashlib
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

# ---------------------------------------------------------------------------
# RUNTIME LOCATION. Rewritten by TASK-2026-09-01-SMCRUCHE-PACKFIX.
#
# It previously did:
#     REPO = os.environ.get("PPSQJ_REPO", <a relative guess>)
#     sys.path.insert(0, os.path.join(
#         REPO, "research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis"))
#     import instrumented as I
#
# That directory is an UNTRACKED local research-task directory. It exists in the
# developer working tree and in NO git clone, so the first ARM 1 Ruche job died
# immediately with `ModuleNotFoundError: No module named 'instrumented'`.
#
# instrumented.py is now BUNDLED, TRACKED and byte-for-byte identical, under
# ../support/, with its SHA256 recorded in ../support/BUNDLE_MANIFEST.json and
# CHECKED below. Nothing else from that untracked directory is needed: the
# transitive import closure of instrumented.py over its siblings is EMPTY -- it
# imports only numpy, dataclasses, time and the TRACKED pps_qj package.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SUPPORT = os.path.abspath(os.path.join(HERE, os.pardir, "support"))

# Repository root: five levels up from <repo>/research/tasks/active/<TASK>/<arm>.
# PPSQJ_REPO overrides it for an unusual layout, but is no longer REQUIRED.
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

# The bundle must be the EXACT file that produced the frozen results. A silent
# substitution here would change the sampler without changing any manifest row.
_man = json.load(open(_need["bundle manifest"]))
for _f in _man["files"]:
    _p = os.path.join(SUPPORT, os.path.basename(_f["bundled_as"]))
    _h = hashlib.sha256(open(_p, "rb").read()).hexdigest()
    if _h != _f["sha256_bundled"]:
        sys.exit("INTEGRITY FAILURE: %s\n  expected sha256 %s\n  found    sha256 %s\n"
                 "  The bundled instrumentation is not the file that produced the "
                 "frozen results. Refusing to run." % (_p, _f["sha256_bundled"], _h))

sys.path.insert(0, REPO)      # the TRACKED pps_qj package
sys.path.insert(0, SUPPORT)   # the TRACKED bundled instrumentation
import instrumented as I
print("[env] python     %s" % sys.executable)
print("[env] instrumented %s" % os.path.abspath(I.__file__))
print("[env] pps_qj     %s" % __import__("pps_qj").__file__)

idx = int(sys.argv[1])
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
cfg = rows[idx]
out_path = os.path.join(outdir, f"{cfg['arm']}_{idx:05d}.json")
if os.path.exists(out_path):
    print(f"[cached] {out_path}"); sys.exit(0)

kw = dict(L=int(cfg["L"]), T=float(cfg["T"]), N_c=int(cfg["N_c"]),
          zeta=float(cfg["zeta"]), lam=float(cfg["lam"]),
          dtau_mult=float(cfg["dtau_mult"]), seed=int(cfg["seed"]),
          resample_scheme=cfg["resample_scheme"])
t0 = time.time()
r = I.run_instrumented(**kw, record_anc=True)
f = np.asarray(r.obs["CMI"], float); w = np.asarray(r.final_weights, float)
ok = np.isfinite(f)
row = dict(cfg)
row.update(status="ok", wall_s=round(time.time() - t0, 2), n_steps=r.n_steps,
           cmi_weighted_mean=float(np.sum(w[ok]*f[ok])/np.sum(w[ok])),
           cmi_unweighted_mean=float(np.mean(f[ok])),
           cmi_within_var=float(np.var(f[ok], ddof=1)),
           n_nonfinite=int((~ok).sum()),
           n_distinct_anc_final=int(np.asarray(r.n_distinct_anc)[-1]),
           gess_final=float(np.asarray(r.gess)[-1]),
           ess_cum_final=float(np.asarray(r.ess_cum)[-1]),
           ess_frac_mean=float(np.mean(np.asarray(r.ess))/int(cfg["N_c"])),
           brentq_fallbacks=int(r.brentq_fallbacks),
           per_clone_CMI=[None if not np.isfinite(x) else round(float(x), 9) for x in f])
json.dump(row, open(out_path, "w"))
print(f"[ok] idx={idx} L={cfg['L']} N_c={cfg['N_c']} wall={row['wall_s']}s -> {out_path}")
