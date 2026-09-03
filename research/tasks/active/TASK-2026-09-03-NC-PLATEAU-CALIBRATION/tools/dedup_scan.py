#!/usr/bin/env python3
"""Duplicate-compute, seed-overlap and reuse-ledger scan.

Four questions, answered against what is ACTUALLY ON DISK rather than against
what any document claims:

  1. Does every entry of build_arms.REUSE match the completed populations that
     really exist at that cell? A reuse ledger that has drifted from the data is
     worse than no ledger: it silently under- or over-counts R.
  2. Does this campaign recompute any (cell, replicate) that an exact-compatible
     population already covers?
  3. Is any seed in this campaign already allocated anywhere in the repository --
     including in the manifests of arms that were never run, which is where a
     collision would otherwise hide?
  4. Does this campaign contain two arms that would compute the same physical
     cell, i.e. is it duplicating itself?

It also writes REUSE_LEDGER.csv, which is the machine-readable answer to (1)
and (2) together.

Read-only over every predecessor directory. Contains no scheduler call.
"""
import os, sys, csv, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
sys.path.insert(0, HERE)
import design as D
from cost_model import n_steps
from build_arms import ARMS, REUSE, norm_cells, build_rows

fail = []
warn = []


def ok(cond, msg):
    print(("  OK    " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


def existing_populations():
    """Every completed exact-compatible population in the repository, by cell."""
    by = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(REPO, "research", "tasks", "**",
                                           "results", "*.json"), recursive=True)):
        if os.path.abspath(p).startswith(TASK + os.sep):
            continue                              # this campaign's own output
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "cmi_weighted_mean" not in d:
            continue
        if d.get("status") not in (None, "ok"):
            continue
        if d.get("resample_scheme", "systematic") != "systematic":
            continue
        parts = os.path.relpath(p, REPO).split(os.sep)
        task = next((x for x in parts if x.startswith("TASK-")), "?")
        arm = parts[parts.index(task) + 1] if task in parts else "?"
        by[(int(d["L"]), float(d["T"]), int(d["N_c"]),
            round(float(d["lam"]), 6), float(d["dtau_mult"]))].append(
            dict(seed=int(d["seed"]), zeta=float(d["zeta"]),
                 src=f"{task}/{arm}", n_steps=int(d["n_steps"]),
                 mean=float(d["cmi_weighted_mean"])))
    return by


def allocated_seeds_everywhere():
    """Every seed named in every manifest.csv and every result JSON, outside
    this campaign. Manifests matter as much as results: an arm that was built
    and never run still owns its seeds."""
    seeds = {}
    for p in glob.glob(os.path.join(REPO, "research", "tasks", "**",
                                    "manifest.csv"), recursive=True):
        if os.path.abspath(p).startswith(TASK + os.sep):
            continue
        for r in csv.DictReader(open(p)):
            seeds[int(r["seed"])] = os.path.relpath(p, REPO)
    for p in glob.glob(os.path.join(REPO, "research", "tasks", "**",
                                    "results", "*.json"), recursive=True):
        if os.path.abspath(p).startswith(TASK + os.sep):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and "seed" in d and "cmi_weighted_mean" in d:
            seeds.setdefault(int(d["seed"]), os.path.relpath(p, REPO))
    return seeds


def main():
    print("=" * 78)
    print("DUPLICATE-COMPUTE AND REUSE SCAN — "
          "TASK-2026-09-03-NC-PLATEAU-CALIBRATION")
    print("=" * 78)

    ex = existing_populations()

    # ---- 1. the reuse ledger against the data ------------------------------
    print("\n1. REUSE LEDGER vs what is on disk")
    for k, (n_claimed, src) in sorted(REUSE.items()):
        have = ex.get(k, [])
        srcs = sorted({h["src"] for h in have})
        ok(len(have) == n_claimed,
           f"L={k[0]:<4} N_c={k[2]:<5} lam={k[3]:<7} dm={k[4]:g}  "
           f"claims {n_claimed}, disk has {len(have)}  {'+'.join(srcs)}")
        if have and src.split("/")[0] not in "".join(srcs):
            warn.append(f"{k}: ledger names {src}, disk says {srcs}")
        for h in have:
            if abs(h["zeta"] - D.ZETA) > 1e-12:
                fail.append(f"{k}: a reused population has zeta={h['zeta']}")
            if h["n_steps"] != n_steps(k[0], k[1], k[3], k[4]):
                fail.append(f"{k}: reused n_steps {h['n_steps']} != "
                            f"{n_steps(k[0], k[1], k[3], k[4])}")
    ok(not warn, "ledger sources agree with disk"
       if not warn else f"source mismatch: {warn}")

    # ---- 2. no recomputation ------------------------------------------------
    print("\n2. NO RECOMPUTATION of an exact-compatible existing population")
    rows_by_arm = {a["name"]: list(csv.DictReader(
        open(os.path.join(TASK, a["name"], "manifest.csv")))) for a in ARMS}
    ledger = []
    for a in ARMS:
        cnt = collections.Counter(
            (int(r["L"]), float(r["T"]), int(r["N_c"]), round(float(r["lam"]), 6),
             float(r["dtau_mult"])) for r in rows_by_arm[a["name"]])
        for k, fresh in sorted(cnt.items()):
            have = len(ex.get(k, []))
            ledger.append(dict(
                arm=a["name"], campaign=a["group"], L=k[0], T=k[1], zeta=D.ZETA,
                lam=k[3], N_c=k[2], dtau_mult=k[4],
                K=n_steps(k[0], k[1], k[3], k[4]),
                target_R=a["R"], existing_reused=have, fresh_here=fresh,
                total_after=have + fresh,
                decision=("top-up" if have else "all fresh"),
                existing_source=REUSE.get(k, (0, ""))[1],
                exact_compatible="yes"))
            ok(have + fresh <= max(a["R"], have),
               f"{a['name']:<20} L={k[0]:<4} N_c={k[2]:<5} lam={k[3]:<7} "
               f"dm={k[4]:g}  {have} existing + {fresh} fresh = "
               f"{have + fresh} (target R={a['R']})")
    # cells that are reused wholesale and appear in NO manifest
    for k, (n, src) in sorted(REUSE.items()):
        if not any(row["L"] == k[0] and row["N_c"] == k[2]
                   and abs(row["lam"] - k[3]) < 1e-9 and row["dtau_mult"] == k[4]
                   for row in ledger):
            ledger.append(dict(
                arm="(none - reused entire)", campaign="-", L=k[0], T=k[1],
                zeta=D.ZETA, lam=k[3], N_c=k[2], dtau_mult=k[4],
                K=n_steps(k[0], k[1], k[3], k[4]), target_R=n,
                existing_reused=n, fresh_here=0, total_after=n,
                decision="reused entire, nothing recomputed",
                existing_source=src, exact_compatible="yes"))
    with open(os.path.join(TASK, "REUSE_LEDGER.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ledger[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(ledger, key=lambda r: (r["campaign"], r["L"],
                                                  r["N_c"], r["lam"])))
    print(f"  wrote REUSE_LEDGER.csv  ({len(ledger)} cell decisions)")

    # ---- 3. seed disjointness ----------------------------------------------
    print("\n3. SEED DISJOINTNESS against the whole repository")
    other = allocated_seeds_everywhere()
    mine = set()
    for a in ARMS:
        mine |= {int(r["seed"]) for r in rows_by_arm[a["name"]]}
    cond = set()
    for p in glob.glob(os.path.join(TASK, "conditional", "*", "manifest.csv")):
        cond |= {int(r["seed"]) for r in csv.DictReader(open(p))}
    clash = (mine | cond) & set(other)
    ok(not clash, f"{len(mine)} immediate + {len(cond)} conditional seeds vs "
                  f"{len(other)} already allocated elsewhere: "
                  f"{'no overlap' if not clash else sorted(clash)[:8]}")
    ok(not (mine & cond), "immediate and conditional blocks are disjoint")
    ok(max(other) < D.SEED_FLOOR,
       f"repository seed ceiling {max(other)} < this campaign's floor "
       f"{D.SEED_FLOOR} (disjointness is STRUCTURAL, not merely observed)")
    ok(all(D.SEED_FLOOR <= s < D.SEED_CEIL for s in mine | cond),
       f"every seed in [{D.SEED_FLOOR}, {D.SEED_CEIL})")

    # ---- 4. self-duplication ------------------------------------------------
    print("\n4. NO SELF-DUPLICATION inside this campaign")
    seen = {}
    dupes = []
    for a in ARMS:
        for r in rows_by_arm[a["name"]]:
            k = (int(r["L"]), float(r["T"]), float(r["zeta"]), round(float(r["lam"]), 6),
                 int(r["N_c"]), float(r["dtau_mult"]))
            if k in seen and seen[k] != a["name"]:
                dupes.append((k, seen[k], a["name"]))
            seen[k] = a["name"]
    ok(not dupes, "no physical cell is built by two different arms"
       if not dupes else f"{dupes[:4]}")
    # the two M96 arms and the two M128 arms ARE the same scan at two N_c and
    # are meant to be mutually exclusive, not duplicates -- flagged, not failed.
    print("  note  conditional/cond_M96_nc1024 vs cond_M96_nc2048 and "
          "cond_M128_nc2048 vs\n        cond_M128_nc4096 are the SAME scan at "
          "two N_c. Exactly one of each pair\n        may ever be released; "
          "their READMEs and interlocks say so.")

    print("\n" + ("SCAN PASSED" if not fail else "SCAN FAILED"))
    for f in fail:
        print("   * " + f)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
