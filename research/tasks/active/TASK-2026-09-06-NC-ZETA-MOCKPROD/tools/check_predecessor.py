#!/usr/bin/env python3
"""Assert TASK-2026-09-06-NC-ZETA-STAGE1 is byte-for-byte as this task found it.

The brief requires the predecessor to remain untouched. This is checked
mechanically rather than promised. Read-only; exits 3 on any difference.
"""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([os.pardir] * 5)))
BASE = os.path.join(HERE, os.pardir, "frozen_inputs", "STAGE1_BASELINE.sha256")
bad = []
n = 0
for line in open(BASE):
    want, path = line.strip().split("  ", 1)
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        bad.append(("MISSING", path, want, "-")); continue
    got = hashlib.sha256(open(full, "rb").read()).hexdigest()
    n += 1
    if got != want:
        bad.append(("MODIFIED", path, want, got))
# an ADDED file is also a modification of the predecessor directory
listed = {l.strip().split("  ", 1)[1] for l in open(BASE)}
pdir = "research/tasks/active/TASK-2026-09-06-NC-ZETA-STAGE1"
for dirpath, _dirs, files in os.walk(os.path.join(ROOT, pdir)):
    for f in files:
        rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
        if rel not in listed:
            bad.append(("ADDED", rel, "-", "-"))
if bad:
    print("PREDECESSOR ISOLATION FAILURE")
    for k, p, w, g in bad:
        print("  %-9s %s\n            expected %s\n            found    %s" % (k, p, w, g))
    sys.exit(3)
print("predecessor isolation OK: %d files verified, none modified, none added" % n)
