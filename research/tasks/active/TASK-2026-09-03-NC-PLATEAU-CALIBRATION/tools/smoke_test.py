#!/usr/bin/env python3
"""End-to-end smoke test of the runtime, at a size that runs in seconds.

What it exercises, in the order a real array task would:

  1. shared/run_cell.py starts, resolves the repository, verifies the bundle
     sha256, imports the certified sampler and pps_qj, and writes a result JSON.
  2. Every field the analysis and the arm QC read is present and of the right
     type -- including the ten per-window histories, final_weights,
     logw_carry_var_final, delta_tau, K, n_resampling_events and git_commit that
     this task added and that no predecessor recorded.
  3. logw_carry_var_final really is Var(log carried weight): recomputed here
     from final_weights independently of the writer.
  4. The result is IDEMPOTENT: a second call on a completed row recomputes
     nothing. That is what makes re-running a partial array safe.
  5. dtau_mult really moves K, with K = ceil(2 lam (L-1) T / dtau_mult) exactly,
     which is campaign E's entire mechanism.
  6. analyse_arm.py runs on the output and refuses to pool two discretisations
     into one cell.
  7. The frozen analysis runs to completion on ZERO new results and says so.

It writes only under a temporary directory inside the task and removes it.
Contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, math, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
PY = os.environ.get("PYTHON", os.path.join(REPO, ".venv", "bin", "python3"))
sys.path.insert(0, HERE)
from cost_model import n_steps

FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult",
          "resample_scheme", "seed"]
fail = []


def ok(cond, msg):
    print(("  OK    " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


def main():
    print("=" * 78)
    print("SMOKE TEST — the real runtime, at toy size")
    print("=" * 78)
    arm = os.path.join(TASK, ".smoke_arm")
    if os.path.isdir(arm):
        shutil.rmtree(arm)
    os.makedirs(os.path.join(arm, "results"))
    for f in ("run_cell.py", "analyse_arm.py"):
        shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(arm, f))

    L, T, lam, N = 12, 12.0, 0.3032, 8
    rows = [dict(arm="SMOKE", L=L, T=T, N_c=N, zeta=0.35, lam=lam,
                 dtau_mult=dm, resample_scheme="systematic",
                 seed=999_000_000 + i)
            for i, dm in enumerate((3.0, 6.0, 12.0))]
    with open(os.path.join(arm, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    try:
        outs = []
        for i in range(len(rows)):
            r = subprocess.run([PY, "run_cell.py", str(i)], cwd=arm,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:])
                ok(False, f"run_cell.py row {i} exited {r.returncode}")
                return 1
            outs.append(json.load(open(os.path.join(
                arm, "results", f"SMOKE_{i:05d}.json"))))
        ok(True, f"{len(outs)} rows executed end to end")

        d = outs[1]                                    # the dtau_mult = 6 row
        need = ["status", "wall_s", "n_steps", "cmi_weighted_mean",
                "cmi_unweighted_mean", "cmi_within_var", "n_nonfinite",
                "n_distinct_anc_final", "gess_final", "ess_cum_final",
                "ess_frac_mean", "brentq_fallbacks", "per_clone_CMI"]
        ok(all(k in d for k in need),
           "every field the PREDECESSOR wrote is still written: "
           + ", ".join(need))
        new = ["delta_tau", "K", "n_resampling_events", "resample_mode",
               "git_commit", "sampler_sha256", "logw_carry_var_final",
               "final_weights", "hist_ess", "hist_ess_cum", "hist_logw_var",
               "hist_w_max", "hist_dLambda_mean", "hist_dLambda_var",
               "hist_n_jumps_mean", "hist_n_distinct_anc", "hist_gess",
               "hist_max_family_frac", "hist_resampled"]
        ok(all(k in d for k in new),
           "every NEW instrumentation field is written: " + ", ".join(new))
        ok(len(d["per_clone_CMI"]) == N and len(d["final_weights"]) == N,
           f"per-clone arrays have length N_c = {N}")
        ok(all(len(d[k]) == d["n_steps"] for k in new if k.startswith("hist_")),
           f"every per-window history has length K = {d['n_steps']}")

        # (3) the accumulated-log-weight spread is what it claims to be
        w = d["final_weights"]
        lw = [math.log(max(x, 1e-300)) for x in w]
        mu = sum(lw) / len(lw)
        var = sum((x - mu) ** 2 for x in lw) / len(lw)
        ok(abs(var - d["logw_carry_var_final"]) < 1e-6,
           f"logw_carry_var_final = Var(log final_weights) recomputed "
           f"independently: {d['logw_carry_var_final']:.6f} vs {var:.6f}   "
           f"(this quantity is recorded in 0 % of the pre-existing corpus)")
        ok(abs(sum(w) - 1.0) < 1e-9, "final_weights are normalised")

        # (4) idempotence
        before = os.path.getmtime(os.path.join(arm, "results", "SMOKE_00000.json"))
        r = subprocess.run([PY, "run_cell.py", "0"], cwd=arm,
                           capture_output=True, text=True)
        after = os.path.getmtime(os.path.join(arm, "results", "SMOKE_00000.json"))
        ok(r.returncode == 0 and before == after and "[cached]" in r.stdout,
           "a completed row is NOT recomputed (re-running a partial array is "
           "safe and tops up rather than redoing)")

        # (5) dtau_mult moves K, exactly
        for row, o in zip(rows, outs):
            want = n_steps(L, T, lam, row["dtau_mult"])
            ok(o["n_steps"] == want and o["K"] == want,
               f"dtau_mult={row['dtau_mult']:<5g} -> K = {o['n_steps']} "
               f"= ceil(2 lam (L-1) T / dtau_mult)")
        ok(len({o["n_steps"] for o in outs}) == 3,
           "the three discretisations really are three different window counts")
        # delta_tau recorded by the sampler is the ACTUAL step T/K, not the
        # nominal dtau_mult/(2 lam (L-1)) it was derived from -- the ceil() in
        # n_steps rounds it down. Recording the nominal value would have been a
        # trap for anyone reconstructing the schedule from a result file.
        ok(all(abs(o["delta_tau"] - T / o["n_steps"]) < 1e-12 for o in outs),
           "delta_tau recorded is the ACTUAL step T/K, not the nominal "
           "dtau_mult/(2 lam (L-1))")
        ok(outs[0]["delta_tau"] < outs[1]["delta_tau"] < outs[2]["delta_tau"],
           "finer dtau_mult really does give a finer step")

        # (6) the arm QC keeps the discretisations apart
        r = subprocess.run([PY, "analyse_arm.py", "results"], cwd=arm,
                           capture_output=True, text=True)
        ok(r.returncode == 0, "analyse_arm.py runs on the output")
        ok(r.stdout.count("dtau_mult=") == 3,
           "analyse_arm.py reports THREE cells, one per discretisation — it "
           "does not pool them")
        ok("Var(log carried weight)" in r.stdout,
           "analyse_arm.py surfaces the new accumulated-weight diagnostic")

        # (7) the frozen analysis survives having nothing new to analyse
        r = subprocess.run(
            [PY, os.path.join(TASK, "analysis", "nc_plateau_analysis.py")],
            capture_output=True, text=True, cwd=REPO)
        ok(r.returncode == 0, "the frozen analysis runs to completion")
        ok("insufficient" in r.stdout or "no populations" in r.stdout,
           "and reports empty sections as empty rather than as a silent pass")
    finally:
        shutil.rmtree(arm, ignore_errors=True)

    print("\n" + ("SMOKE TEST PASSED" if not fail else "SMOKE TEST FAILED"))
    for f in fail:
        print("   * " + f)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
