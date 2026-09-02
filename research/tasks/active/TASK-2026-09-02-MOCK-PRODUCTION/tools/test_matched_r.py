#!/usr/bin/env python3
"""Unit checks for the matched-R amendment (../MATCHED_R_AMENDMENT.md).

The amendment's whole value rests on one property: block membership is decided
by SEED ORDER ALONE and can never depend on the observable. If that fails, the
"primary" subset becomes a choice made after seeing the data, which is exactly
what the amendment exists to prevent. So it is asserted here rather than
assumed, on synthetic cells that need no campaign data.

    python3 tools/test_matched_r.py

Read-only. Writes nothing. Contains no scheduler call.
"""
import csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(TASK, "analysis"))
import mock_production_analysis as A          # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_cell(seeds, pops, N_c=1024):
    """A cell as load() would leave it: sorted by seed, stats computed."""
    seeds = np.asarray(seeds, dtype=np.int64)
    pops = np.asarray(pops, float)
    order = np.argsort(seeds, kind="stable")
    c = dict(seeds=seeds[order], pops=pops[order],
             within=np.full(pops.size, 0.05)[order],
             anc=np.full(pops.size, 2.0)[order],
             wall=np.full(pops.size, 100.0)[order],
             nonfin=0, clones=N_c * pops.size, fallbacks=0,
             src={"synthetic"}, N_c=N_c, lam=0.3032)
    c.update(A._stats(c["pops"], c["within"], N_c))
    c["n_blocks"] = c["R"] // A.BLOCK
    c["block"] = None
    return c


print("=" * 78)
print("  matched-R block-selection unit checks")
print("=" * 78)

# ---------------------------------------------------------------- 1
# BLOCK SELECTION IS OBSERVABLE-BLIND.
# Same seeds, wildly different CMI values (including a monotone-sorted set and
# a reversed one). The seed membership of every block must be byte-identical.
rng = np.random.default_rng(1)
seeds = np.arange(30300000, 30300096)
variants = {
    "random":        rng.normal(0.4, 0.05, 96),
    "sorted asc":    np.sort(rng.normal(0.4, 0.05, 96)),
    "sorted desc":   np.sort(rng.normal(0.4, 0.05, 96))[::-1],
    "adversarial":   np.where(np.arange(96) < 24, 99.0, -99.0),
    "all identical": np.full(96, 0.4),
}
ref = None
same = True
for name, pops in variants.items():
    c = make_cell(seeds, pops)
    memb = [tuple(A.cell_block(c, b)["seeds"].tolist()) for b in range(4)]
    if ref is None:
        ref = memb
    elif memb != ref:
        same = False
        print(f"        block membership CHANGED for variant {name!r}")
check("block membership is identical across 5 CMI variants of the same seeds",
      same, f"({len(variants)} variants, 4 blocks each)")

# and it is the seed order, not the file order: shuffle the input order
shuf = rng.permutation(96)
c_shuf = make_cell(seeds[shuf], variants["random"][shuf])
c_plain = make_cell(seeds, variants["random"])
check("block membership is invariant to input row order",
      all(np.array_equal(A.cell_block(c_shuf, b)["seeds"],
                         A.cell_block(c_plain, b)["seeds"]) for b in range(4)))
check("and the paired CMI values travel with their seeds",
      all(np.allclose(A.cell_block(c_shuf, b)["pops"],
                      A.cell_block(c_plain, b)["pops"]) for b in range(4)))

# ---------------------------------------------------------------- 2
# EXACT SPLITS.
c96 = make_cell(seeds, variants["random"])
check("R=96 gives exactly 4 full blocks", c96["n_blocks"] == 4,
      f"n_blocks={c96['n_blocks']}")
check("R=96 blocks are 24+24+24+24",
      [A.cell_block(c96, b)["R"] for b in range(4)] == [24, 24, 24, 24])
check("R=96 has no 5th block", A.cell_block(c96, 4) is None)

c48 = make_cell(np.arange(31400000, 31400048), rng.normal(0.4, 0.05, 48), N_c=128)
check("R=48 gives exactly 2 full blocks", c48["n_blocks"] == 2)
check("R=48 blocks are 24+24",
      [A.cell_block(c48, b)["R"] for b in range(2)] == [24, 24])
check("R=48 has no 3rd block", A.cell_block(c48, 2) is None)

c24 = make_cell(np.arange(31000000, 31000024), rng.normal(0.4, 0.05, 24))
check("R=24 gives exactly 1 full block", c24["n_blocks"] == 1)
check("R=24 block A is the whole cell",
      A.cell_block(c24, 0)["R"] == 24 and A.cell_block(c24, 1) is None)

c23 = make_cell(np.arange(31000000, 31000023), rng.normal(0.4, 0.05, 23))
check("R=23 yields NO full block (a short cell cannot masquerade as matched)",
      c23["n_blocks"] == 0 and A.cell_block(c23, 0) is None)

# ---------------------------------------------------------------- 3
# DISJOINTNESS AND COVERAGE.
allseeds = []
for b in range(4):
    allseeds += A.cell_block(c96, b)["seeds"].tolist()
check("the 4 blocks of an R=96 cell are pairwise disjoint",
      len(set(allseeds)) == 96, f"{len(set(allseeds))} distinct of 96")
check("the 4 blocks together cover the whole cell exactly",
      sorted(allseeds) == sorted(seeds.tolist()))
b_seeds = [set(A.cell_block(c48, b)["seeds"].tolist()) for b in range(2)]
check("the 2 blocks of an R=48 cell are disjoint and cover it",
      not (b_seeds[0] & b_seeds[1]) and len(b_seeds[0] | b_seeds[1]) == 48)

# ---------------------------------------------------------------- 4
# BLOCKS ARE CONSECUTIVE IN SEED ORDER, and A is the lowest.
check("block A holds the 24 lowest seeds",
      A.cell_block(c96, 0)["seeds"].tolist() == sorted(seeds)[:24])
check("blocks are consecutive and ascending in seed order",
      all(A.cell_block(c96, b)["seeds"].max() < A.cell_block(c96, b + 1)["seeds"].min()
          for b in range(3)))
check("PRIMARY_BLOCK is block A", A.PRIMARY_BLOCK == 0 and
      A.BLOCK_LABELS[A.PRIMARY_BLOCK] == "A")

# ---------------------------------------------------------------- 5
# THE BLOCK MEAN IS THE MEAN OF THAT BLOCK, not of the parent.
bA = A.cell_block(c96, 0)
check("block statistics are computed from the block, not inherited",
      np.isclose(bA["mean"], c96["pops"][:24].mean())
      and not np.isclose(bA["mean"], c96["mean"]),
      f"block A mean {bA['mean']:.6f} vs parent {c96['mean']:.6f}")
check("block SEM uses the block's own R",
      np.isclose(bA["sem"], c96["pops"][:24].std(ddof=1) / np.sqrt(24)))

# ---------------------------------------------------------------- 6
# THE REAL MANIFESTS SUPPLY THE R THE AMENDMENT ASSUMES.
print("\n  manifest R per (arm, lambda) — what the campaign will actually return:")
expect = {"mockL32": 24, "mockL48": 24, "mockL64": 24, "mockL64nc2048": 24,
          "mockNC128L32": 48, "mockNC128L48": 48, "mockNC128L64": 48}
ok_all = True
for arm, want in sorted(expect.items()):
    p = os.path.join(TASK, arm, "manifest.csv")
    if not os.path.isfile(p):
        check(f"{arm} manifest present", False); ok_all = False; continue
    rows = list(csv.DictReader(open(p)))
    per = {}
    for r in rows:
        per[round(float(r["lam"]), 6)] = per.get(round(float(r["lam"]), 6), 0) + 1
    Rs = sorted(set(per.values()))
    good = Rs == [want] and want % A.BLOCK == 0
    ok_all &= good
    print(f"    {'PASS' if good else 'FAIL'}  {arm:<16} R = {Rs}, "
          f"{want // A.BLOCK} full block(s) of {A.BLOCK}")
    if not good:
        FAILS.append(f"{arm} R")
check("every arm's R is an exact multiple of the matched block size", ok_all)

# the reused ARM-B frozen input
fp = os.path.join(TASK, "frozen_inputs", "armB_populations.csv")
if os.path.isfile(fp):
    per = {}
    for r in csv.DictReader(open(fp)):
        if r["status"] == "ok":
            per.setdefault(round(float(r["lam"]), 6), []).append(int(r["seed"]))
    Rs = sorted({len(v) for v in per.values()})
    check("reused ARM-B cells hold R=96, i.e. exactly 4 blocks of 24",
          Rs == [96], f"R per lambda = {Rs}, lambdas = {len(per)}")
    check("reused ARM-B seeds are unique within every cell",
          all(len(set(v)) == len(v) for v in per.values()))
else:
    check("frozen ARM-B input present", False)

print("\n" + "=" * 78)
print(f"  {len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else " — all checks pass"))
print("=" * 78)
sys.exit(1 if FAILS else 0)
