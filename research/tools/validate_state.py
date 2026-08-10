#!/usr/bin/env python3
"""validate_state.py - integrity checks for the ppsQJ_m2 knowledge plane.

Read-only. Modifies nothing. Exit 0 clean, 1 if any ERROR.

Checks:
  E1  duplicate entity IDs
  E2  id does not match filename
  E3  dangling reference (claim -> evidence/observable/claim/dispute/decision)
  E4  'supported' claim whose evidence is not reproducible
  E5  'supported' claim with no discriminating evidence
  E13 missing/invalid statement_class                          (charter sec 2)
  E14 missing/invalid claim_kind                               (charter Stage 7)
  E15 missing/invalid epistemic_status                         (charter Stage 7)
  E16 missing/invalid confidence, or missing confidence_basis  (charter Stage 7)
  E17 judgment asserted as 'supported'                         (charter sec 2)
  E18 conjecture asserted as 'supported'                       (charter sec 2)
  E19 'supported' backed only by chat_only/unrecoverable evidence
  E20 uninspected source carrying a 'supported' claim          (charter sec 4.1)
  E21 data-root registry missing or unparseable
  E22 evidence references an undeclared logical data root
  E6  claim depending on a refuted/superseded claim -> STALE   (charter sec 8)
  E7  claim referencing a superseded claim as live evidence
  E8  broken local path where the schema asserts exists: true
  E9  contested claim without contests[] or dispute_id
  E10 exponent/amplitude claim missing parameterization or observable_id
  E11 claim with empty evidence and no evidence_note
  E12 asymmetric contests link

Usage:  python3 research/tools/validate_state.py [--repo ROOT] [-q]
"""
from __future__ import annotations
import argparse, os, re, sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  .venv/bin/python3 -m pip install pyyaml")

KINDS = ("claims", "evidence", "observables", "sources", "decisions", "disputes")
# Charter Stage 7 vocabulary is authoritative.
STATUS = {"unsupported", "provisional", "supported", "contradicted", "withdrawn"}
DEAD_STATUS = {"contradicted", "withdrawn"}
STATEMENT_CLASS = {"evidence", "inference", "conjecture", "judgment"}
CLAIM_KIND = {"theorem", "empirical_result", "interpretation", "conjecture", "judgment",
              "derivation", "methodological_result", "negative_result",
              "data_adequacy_assessment", "infrastructure_fact"}
CONFIDENCE = {"unassessed", "very_low", "low", "moderate", "high", "very_high"}
WEAK_REPRO = {"chat_only", "unrecoverable", "memory"}
OK_REPRO = {"fully_reproducible", "partially_reproducible"}
NEEDS_PARAM = {"exponent", "amplitude", "scaling_law"}

errors: list[str] = []
warns: list[str] = []


def err(c, f, m): errors.append(f"ERROR {c} [{f}] {m}")
def warn(c, f, m): warns.append(f"WARN  {c} [{f}] {m}")


def load(root):
    ents, byid = {k: {} for k in KINDS}, {}
    for k in KINDS:
        d = os.path.join(root, "research", "state", k)
        if not os.path.isdir(d):
            warn("W0", k, "directory missing"); continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith((".yaml", ".yml")):
                continue
            p = os.path.join(d, fn)
            try:
                doc = yaml.safe_load(open(p, encoding="utf-8")) or {}
            except Exception as e:
                err("E0", fn, f"YAML parse failure: {e}"); continue
            if not isinstance(doc, dict):
                err("E0", fn, "top level is not a mapping"); continue
            i = doc.get("id")
            if not i:
                err("E2", fn, "missing id"); continue
            if i in byid:
                err("E1", fn, f"duplicate id {i} (also in {byid[i][0]})")
            if i != os.path.splitext(fn)[0]:
                err("E2", fn, f"id {i} does not match filename")
            doc["_file"], doc["_kind"] = fn, k
            ents[k][i] = doc
            byid[i] = (fn, doc)
    return ents, byid


def ev_ids(claim):
    """evidence: may be [ID] or [{id: ID, role: ...}]"""
    out = []
    for e in claim.get("evidence") or []:
        out.append(e.get("id") if isinstance(e, dict) else e)
    return [x for x in out if x]


def ev_roles(claim):
    return {e.get("id"): e.get("role") for e in (claim.get("evidence") or [])
            if isinstance(e, dict)}


def refs_in(node, pat=re.compile(
        r"(?<![\w-])(?:CB|CASEA|VR|METH|INFRA|PPS|EV|OBS|SRC|DEC|DISP|EXP)"
        r"(?:-[A-Z0-9]+)+(?![\w-])")):
    """every ID-looking token appearing in reference FIELDS only"""
    found = set()
    def walk(v):
        if isinstance(v, str):
            found.update(pat.findall(v))
        elif isinstance(v, list):
            for x in v: walk(x)
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
    for key in ("depends_on", "contests", "supersedes", "superseded_by",
                "dispute_id", "observable_id", "competing_claims",
                "input_evidence", "supports", "contradicts", "claim_ref",
                "depends_on_claims", "blocking_gaps"):
        if key in node: walk(node[key])
    return found


def load_local_roots(repo):
    """Machine-local logical-root mapping. Absent file is NOT an error."""
    p = os.path.join(repo, "research", "data_roots.local.yaml")
    if not os.path.exists(p):
        warn("W1", "data_roots.local.yaml",
             "absent; {root,relative} references cannot be resolved on this machine")
        return {}
    try:
        d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    except Exception as e:
        err("E21", "data_roots.local.yaml", f"parse failure: {e}")
        return {}
    return d.get("roots") or {}


def known_roots(repo):
    p = os.path.join(repo, "research", "state", "DATA_ROOTS.yaml")
    try:
        d = yaml.safe_load(open(p, encoding="utf-8")) or {}
        return set((d.get("roots") or {}).keys())
    except Exception:
        err("E21", "DATA_ROOTS.yaml", "missing or unparseable")
        return set()


def check_paths(repo, doc, f, local, declared):
    unresolved = []

    def walk(v):
        if isinstance(v, dict):
            # portable form: {root: LOGICAL, relative: rel/path, exists: true}
            if "root" in v and "relative" in v:
                r = v["root"]
                if r not in declared:
                    err("E22", f, f"unknown logical root {r!r}; "
                                 f"declare it in research/state/DATA_ROOTS.yaml")
                elif str(v.get("exists")).lower() == "true":
                    base = local.get(r)
                    if base is None:
                        unresolved.append(f"{r}:{v['relative']}")
                    else:
                        ap = os.path.join(base, v["relative"])
                        if not os.path.exists(ap):
                            err("E8", f, f"path asserted to exist is missing: "
                                         f"{r}/{v['relative']}  (-> {ap})")
            # legacy absolute/repo-relative form, still supported
            elif "path" in v and isinstance(v["path"], str):
                if str(v.get("exists")).lower() == "true":
                    pth = v["path"]
                    ap = pth if os.path.isabs(pth) else os.path.join(repo, pth)
                    if not os.path.exists(ap):
                        err("E8", f, f"path asserted to exist is missing: {pth}")
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)

    walk(doc)
    for u in unresolved:
        warn("W2", f, f"unresolved on this machine (root not mapped): {u}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.repo)

    ents, byid = load(root)
    claims, evid = ents["claims"], ents["evidence"]

    for i, c in claims.items():
        f, st = c["_file"], c.get("epistemic_status")

        # E3 dangling references
        for r in refs_in(c):
            if r not in byid:
                err("E3", f, f"dangling reference {r}")

        # E11 evidence present or explicitly excused
        eids = ev_ids(c)
        if not eids and not c.get("evidence_note"):
            err("E11", f, "empty evidence and no evidence_note")

        roles = ev_roles(c)

        # E13 charter taxonomy present and valid
        sc, ck, cf = c.get("statement_class"), c.get("claim_kind"), c.get("confidence")
        if sc not in STATEMENT_CLASS:
            err("E13", f, f"statement_class missing/invalid: {sc!r} (charter sec 2)")
        if ck not in CLAIM_KIND:
            err("E14", f, f"claim_kind missing/invalid: {ck!r} (charter Stage 7)")
        if st not in STATUS:
            err("E15", f, f"epistemic_status missing/invalid: {st!r} (charter Stage 7)")
        if cf not in CONFIDENCE:
            err("E16", f, f"confidence missing/invalid: {cf!r} (charter Stage 7)")
        elif not c.get("confidence_basis"):
            err("E16", f, "confidence set but confidence_basis missing")

        # E17 a judgment is not a fact (charter sec 2)
        if sc == "judgment" and st == "supported":
            err("E17", f, "statement_class judgment may not be 'supported'")
        # E18 a conjecture is by definition unsupported (charter sec 2)
        if ck == "conjecture" and st == "supported":
            err("E18", f, "claim_kind conjecture may not be 'supported'")

        if st == "supported":
            # E5 discriminating evidence required
            disc = [k for k, v in roles.items() if v == "discriminating"]
            if not disc and not c.get("review_record", {}).get("notes"):
                err("E5", f, "supported with no discriminating evidence")
            # E4 reproducibility floor
            strong = False
            for e in eids:
                ed = evid.get(e)
                if ed and ed.get("reproducibility") in OK_REPRO:
                    strong = True
            if eids and not strong:
                err("E4", f, "supported but no supporting evidence is "
                             "fully/partially reproducible")
            # E19 weak-evidence-only cap
            if eids and all((evid.get(e) or {}).get("reproducibility") in WEAK_REPRO
                            for e in eids if e in evid):
                err("E19", f, "supported but all evidence is chat_only/unrecoverable; "
                              "charter caps this at 'provisional'")

        # E6 dependency cascade
        for d in c.get("depends_on") or []:
            dd = claims.get(d)
            if dd and dd.get("epistemic_status") in DEAD_STATUS:
                if c.get("review_status") != "stale":
                    err("E6", f, f"depends on {d} (epistemic_status={dd['epistemic_status']}) "
                                 f"but review_status is "
                                 f"'{c.get('review_status')}', expected 'stale'")

        # E7 superseded claim cited as live evidence
        for e in eids:
            if e in claims and claims[e].get("epistemic_status") in DEAD_STATUS:
                err("E7", f, f"cites {e} which is {claims[e]['epistemic_status']}")

        # E9 contested bookkeeping (now an orthogonal boolean, not a status)
        if c.get("contested") is True:
            if not c.get("contests"):
                err("E9", f, "contested: true but contests[] empty")
            if not c.get("dispute_id"):
                err("E9", f, "contested: true but no dispute_id")
        if c.get("contests") and c.get("contested") is not True:
            err("E9", f, "has contests[] but contested is not true")

        # E10 well-formedness of numeric claims
        if c.get("type") in NEEDS_PARAM:
            if not c.get("parameterization"):
                err("E10", f, f"type {c['type']} requires parameterization")
            if not c.get("observable_id"):
                # Charter: an entry that cannot be classified confidently must be
                # marked EXPLICITLY unresolved rather than guessed. A declared
                # gap is a warning; a silent omission is an error.
                unres = c.get("observable_id_unresolved")
                if isinstance(unres, dict) and unres.get("candidates") and unres.get("reason"):
                    warn("W3", f, f"observable_id declared UNRESOLVED "
                                  f"(candidates: {unres['candidates']}) - "
                                  f"claim cannot be promoted until settled")
                else:
                    err("E10", f, f"type {c['type']} requires observable_id, or an "
                                  f"observable_id_unresolved block with candidates and reason")

        # E12 symmetric contests
        for o in c.get("contests") or []:
            od = claims.get(o)
            if od and i not in (od.get("contests") or []):
                err("E12", f, f"contests {o} but {o} does not contest back")

    # E20 charter sec 4.1: a source not actually inspected cannot carry a
    # 'supported' claim. A title/snippet match is not evidence.
    OK_INSPECT = {"fully_inspected", "relevant_sections"}
    for si, sd in ents["sources"].items():
        lvl = sd.get("inspection_level")
        if lvl is None:
            err("E20", sd["_file"], "source missing inspection_level (charter sec 4.1)")
            continue
        if lvl in OK_INSPECT:
            continue
        for cid in sd.get("invoked_for_claims") or []:
            cd = claims.get(cid)
            if cd and cd.get("epistemic_status") == "supported":
                err("E20", sd["_file"],
                    f"claim {cid} is 'supported' but leans on this source at "
                    f"inspection_level={lvl} (charter sec 4.1)")

    local = load_local_roots(root)
    declared = known_roots(root)
    for kind in KINDS:
        for i, d in ents[kind].items():
            check_paths(root, d, d["_file"], local, declared)
            for r in refs_in(d):
                if r not in byid:
                    err("E3", d["_file"], f"dangling reference {r}")

    n = sum(len(ents[k]) for k in KINDS)
    if not a.quiet:
        print(f"scanned {n} entities: " +
              ", ".join(f"{k}={len(ents[k])}" for k in KINDS))
    for w in warns: print(w)
    for e in errors: print(e)
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
