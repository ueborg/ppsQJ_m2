#!/usr/bin/env python3
"""resolve_provenance.py - follow a claim's DIRECT load-bearing references.

Read-only. Given claim/evidence/observable IDs, emit the references they point
at directly: cited evidence, depends_on, supersedes/superseded_by, contests,
observable_id, dispute_id, and - critically - any FILE PATH named in a prose
provenance field.

    resolve_provenance.py CB-AMP-096-001 [CB-AMP-001 ...] [--json]

WHY THIS EXISTS
In TASK-2026-08-10-AMP096, candidate C1 asserted that A = 0.96 "is not and never
was an r_c-type prefactor". CB-AMP-096-001's own provenance_note says, in plain
text, that the number WAS reinterpreted as an r_c prefactor on 2026-06-10 and
names the document that did it. Three workers read the claim and none followed
the reference. The red team did, and killed the candidate on it.

The failure was not a lack of history: it was a lack of ONE HOP. So this tool
does exactly one hop from the claims in scope, and stops. It does NOT recursively
preload project history - that is the opposite failure and it is expensive.

Files surfaced here are PROVENANCE, never support (Charter Appendix A.3). Their
value is that they record what the project previously believed, which is
precisely what you need before declaring a claim's history false.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required")

KINDS = ("claims", "evidence", "observables", "sources", "decisions", "disputes")

ID_FIELDS = ("depends_on", "supersedes", "superseded_by", "contests",
             "dispute_id", "observable_id", "observable_id_previous",
             "input_evidence", "supports", "contradicts", "distinct_from",
             "invoked_for_claims", "competing_claims", "source_of_definition")

# Prose fields that historically carry a pointer to the document that corrected,
# reinterpreted or generated the claim.
PROSE_FIELDS = ("provenance_note", "confidence_basis", "evidence_note",
                "retirement_note", "caveat", "note", "notes", "rationale",
                "audit_action_required", "recoverable_from", "why_it_matters",
                "scope", "open_task")

# A path-like token: at least one directory separator or a known extension.
PATH_RE = re.compile(
    r"\b((?:[\w.\-]+/)+[\w.\-]+\.(?:md|py|yaml|yml|csv|json|tex|pkl|npz|sh|ipynb)"
    r"|[\w.\-]+\.(?:md|py|yaml|yml|csv|tex|ipynb))\b")
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def locate(root, basename, limit=4):
    """Find a bare filename somewhere in the repo. Returns relative paths."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if basename in filenames:
            hits.append(os.path.relpath(os.path.join(dirpath, basename), root))
            if len(hits) >= limit:
                break
    return hits or None


def load_state(root):
    byid = {}
    for kind in KINDS:
        for p in sorted(glob.glob(os.path.join(root, "research", "state", kind, "*.yaml"))):
            try:
                d = yaml.safe_load(open(p, encoding="utf-8")) or {}
            except Exception:
                continue
            if isinstance(d, dict) and d.get("id"):
                d["_kind"], d["_path"] = kind, os.path.relpath(p, root)
                byid[d["id"]] = d
    return byid


def walk_ids(node, out, depth=0):
    if depth > 5:
        return
    if isinstance(node, str):
        out.update(re.findall(
            r"\b(?:CB|CASEA|VR|METH|INFRA|PPS|EV|OBS|SRC|DEC|DISP|EXP)"
            r"(?:-[A-Z0-9]+)+\b", node))
    elif isinstance(node, list):
        for v in node:
            walk_ids(v, out, depth + 1)
    elif isinstance(node, dict):
        for v in node.values():
            walk_ids(v, out, depth + 1)


def prose_of(doc, depth=0):
    out = []
    if depth > 4:
        return out
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in PROSE_FIELDS and isinstance(v, str):
                out.append((k, v))
            elif isinstance(v, (dict, list)):
                out.extend(prose_of(v, depth + 1))
    elif isinstance(doc, list):
        for v in doc:
            out.extend(prose_of(v, depth + 1))
    return out


def resolve(root, ids):
    byid = load_state(root)
    report = []
    for cid in ids:
        doc = byid.get(cid)
        if doc is None:
            report.append({"id": cid, "error": "not found in research/state/**"})
            continue

        refs = set()
        for f in ID_FIELDS:
            if f in doc:
                walk_ids(doc[f], refs)
        for e in doc.get("evidence") or []:
            refs.add(e.get("id") if isinstance(e, dict) else e)
        refs.discard(None)
        refs.discard(cid)

        docs_named = []
        for field, text in prose_of(doc):
            for m in PATH_RE.findall(text):
                ap = os.path.join(root, m)
                exists = os.path.exists(ap)
                located = None
                if not exists:
                    # Prose often names a bare filename ("Y_ZETA_DERIVATION.md")
                    # rather than a path. Locating it is the whole point: an
                    # unresolved reference is a reference nobody follows.
                    located = locate(root, os.path.basename(m))
                # a nearby date is usually the correction date; keep it, it is
                # what tells you WHICH of two accounts is the later one
                dates = DATE_RE.findall(text)
                docs_named.append({
                    "path": m, "field": field, "exists": exists,
                    "located_at": located,
                    "dates_in_field": dates,
                    "excerpt": text.strip()[:400],
                })

        report.append({
            "id": cid,
            "kind": doc["_kind"],
            "file": doc["_path"],
            "status": doc.get("epistemic_status") or doc.get("status"),
            "statement": (doc.get("statement") or doc.get("name") or "")[:200],
            "referenced_ids": sorted(
                {r: None for r in refs}),
            "unresolvable_ids": sorted(r for r in refs if r not in byid),
            "documents_named_in_prose": docs_named,
        })
    return report


BANNER = """
DIRECT PROVENANCE - one hop, no recursion.

Everything below is PROVENANCE, not support. You may not cite it as why a thing
is true. You MUST read it before asserting that this claim's own account of its
history is wrong: the account may be recorded in a document the claim names.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.repo)
    rep = resolve(root, a.ids)

    if a.json:
        print(json.dumps(rep, indent=2))
        return 0

    print(BANNER)
    for r in rep:
        if r.get("error"):
            print(f"{r['id']}: {r['error']}")
            continue
        print(f"=== {r['id']}  [{r['kind'][:-1]}, {r['status']}]  {r['file']}")
        print(f"    {r['statement']}")
        if r["referenced_ids"]:
            print(f"    references: {', '.join(r['referenced_ids'])}")
        if r["unresolvable_ids"]:
            print(f"    UNRESOLVABLE: {', '.join(r['unresolvable_ids'])}")
        if r["documents_named_in_prose"]:
            print("    DOCUMENTS NAMED IN PROSE - read these before contradicting "
                  "this claim's history:")
            for d in r["documents_named_in_prose"]:
                if d["exists"]:
                    mark = "exists"
                elif d.get("located_at"):
                    mark = "named without a path; FOUND AT " + ", ".join(d["located_at"])
                else:
                    mark = "MISSING ON DISK"
                dates = f", dates: {', '.join(d['dates_in_field'])}" if d["dates_in_field"] else ""
                print(f"      - {d['path']}  ({mark}{dates})   [{d['field']}]")
                print(f"        \"{d['excerpt'][:220]}\"")
        else:
            print("    (no documents named in prose fields)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
