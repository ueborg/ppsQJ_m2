#!/usr/bin/env python
"""Block-pooling aggregator for the N_c-ladder campaign.

The stock aggregate_clone keys by (L, lam, zeta) and OVERWRITES on duplicate
keys, so it cannot combine the ladder's seed-blocks (multiple 5-realisation
tasks per point).  This pools them: for each (L, lam, zeta) it concatenates
every per-realisation array (*_means_all) across all npz in the directory and
recomputes mean / std / sem over the pooled seeds.

Run ONCE PER RUNG DIRECTORY (each dir is a single N_c):

    python analysis/aggregate_ladder.py /scratch/$USER/pps_qj/pps_clone_ladder_nc500 \
        --out /scratch/$USER/pps_qj/ladder_nc500.pkl

Output: pkl keyed (L, round(lam,4), round(zeta,3)) -> rec with pooled scalars
(B_L_mean/err, CMI_mean/err, S_mean/err, S_renyi_2/3_mean, S_AB_mean, corr) and
an n_seeds field.  Drop-in for nc_bias_pairs / debias_collapse / scaling_form.
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# per-realisation arrays in the npz -> (pooled scalar mean name, pooled err name)
ARR = {
    "S_means_all":      ("S_mean", "S_err"),
    "B_L_means_all":    ("B_L_mean", "B_L_err"),
    "CMI_means_all":    ("CMI_mean", "CMI_err"),
    "S_AB_means_all":   ("S_AB_mean", "S_AB_err"),
    "S_BC_means_all":   ("S_BC_mean", "S_BC_err"),
    "S_B_means_all":    ("S_B_mean", "S_B_err"),
    "S_ABC_means_all":  ("S_ABC_mean", "S_ABC_err"),
    "S_renyi_2s_all":   ("S_renyi_2_mean", "S_renyi_2_err"),
    "S_renyi_3s_all":   ("S_renyi_3_mean", "S_renyi_3_err"),
    "thetas_all":       ("theta_mean", "theta_err"),
}
SCALAR_COPY = ("L", "lam", "zeta", "N_c", "T")


def _key(L, lam, zeta):
    return (int(L), round(float(lam), 4), round(float(zeta), 3))


def _sem(a):
    a = a[np.isfinite(a)]
    n = a.size
    return (float(np.mean(a)), float(np.std(a)), float(np.std(a) / np.sqrt(n)) if n else float("nan"), n)


def aggregate_ladder(output_dir: Path) -> dict:
    files = sorted(Path(output_dir).glob("clone_*.npz"))
    # group npz by (L,lam,zeta)
    groups = defaultdict(list)
    for f in files:
        try:
            d = dict(np.load(f, allow_pickle=False))
        except Exception:
            continue
        if "B_L_means_all" not in d and "S_means_all" not in d:
            continue
        groups[_key(d["L"], d["lam"], d["zeta"])].append(d)

    out = {}
    for key, recs in groups.items():
        rec = {}
        for s in SCALAR_COPY:
            for d in recs:
                if s in d:
                    rec[s] = float(np.asarray(d[s]).ravel()[0]); break
        # N_c consistency check within a key (should be one rung per dir)
        ncs = {int(np.asarray(d["N_c"]).ravel()[0]) for d in recs if "N_c" in d}
        if len(ncs) > 1:
            print(f"  WARNING {key}: mixed N_c {ncs} in one dir — pooling anyway", file=sys.stderr)
        # pool each per-realisation array across blocks
        n_seeds = 0
        for arr_name, (mname, ename) in ARR.items():
            pooled = []
            for d in recs:
                if arr_name in d:
                    pooled.append(np.asarray(d[arr_name], dtype=float).ravel())
            if pooled:
                allv = np.concatenate(pooled)
                m, sd, sem, n = _sem(allv)
                rec[mname] = m; rec[ename] = sem
                if arr_name == "B_L_means_all":
                    n_seeds = n
        if n_seeds == 0:  # fall back to S count
            for d in recs:
                if "S_means_all" in d:
                    n_seeds += int(np.sum(np.isfinite(np.asarray(d["S_means_all"], float))))
        rec["n_seeds"] = int(n_seeds)
        rec["n_blocks"] = len(recs)
        # corr_decay: average the per-task means (same length per point)
        cds = [np.asarray(d["corr_decay_mean"], float) for d in recs
               if "corr_decay_mean" in d and np.asarray(d["corr_decay_mean"]).size]
        if cds:
            L = min(len(c) for c in cds)
            rec["corr_decay_mean"] = np.mean(np.stack([c[:L] for c in cds]), axis=0)
            for d in recs:
                if "corr_decay_r" in d and np.asarray(d["corr_decay_r"]).size:
                    rec["corr_decay_r"] = np.asarray(d["corr_decay_r"], float)[:L]; break
        out[key] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pool seed-blocks for one N_c rung dir")
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    data = aggregate_ladder(Path(a.dir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(data, f, protocol=4)
    Ls = sorted({k[0] for k in data}); zs = sorted({k[2] for k in data})
    ns = [r["n_seeds"] for r in data.values()]
    ncs = sorted({r.get("N_c") for r in data.values()})
    print(f"pooled {len(data)} points  L={Ls}  zeta={zs}  N_c={ncs}")
    print(f"seeds/point: min={min(ns)} median={int(np.median(ns))} max={max(ns)}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
