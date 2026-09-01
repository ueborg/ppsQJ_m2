#!/usr/bin/env python3
"""Execute ONE manifest row. Called once per array task.

Runs the CERTIFIED production path (pps_qj) through the parent's instrumented
wrapper, which TASK-2026-08-30-SMCSTAT validated bitwise against production.
Writes one JSON row. Idempotent: a completed row is never recomputed.
"""
import os, sys, json, csv, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PPSQJ_REPO", os.path.abspath(os.path.join(HERE, "../../../..")))
sys.path.insert(0, os.path.join(REPO, "research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis"))
import instrumented as I

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
