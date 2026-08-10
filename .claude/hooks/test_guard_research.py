#!/usr/bin/env python3
"""Self-test for guard_research.py. Run:

    .venv/bin/python3 .claude/hooks/test_guard_research.py

Exit 0 = every invariant holds. This is infrastructure, so it is tested; a
guard that silently stops guarding is worse than no guard, because the prose
still claims protection.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard_research.py")
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(HOOK), "..", ".."))


def run(tool_name, tool_input, env_extra=None):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=PROJECT_DIR)
    env.pop("PPSQJ_ALLOW_STATE_WRITE", None)
    env.update(env_extra or {})
    payload = json.dumps({"hook_event_name": "PreToolUse",
                          "tool_name": tool_name, "tool_input": tool_input})
    p = subprocess.run([sys.executable, HOOK], input=payload,
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return "ERROR", p.stderr.strip()
    if not p.stdout.strip():
        return "allow", ""
    out = json.loads(p.stdout)["hookSpecificOutput"]
    return out["permissionDecision"], out["permissionDecisionReason"]


CASES = [
    # (label, tool, input, expected decision)
    # --- G1 state protection -------------------------------------------------
    ("G1 Edit a claim", "Edit",
     {"file_path": "research/state/claims/CB-AMP-001.yaml"}, "deny"),
    ("G1 Write a new claim", "Write",
     {"file_path": "research/state/claims/NEW-001.yaml"}, "deny"),
    ("G1 absolute path", "Edit",
     {"file_path": f"{PROJECT_DIR}/research/state/evidence/EV-X.yaml"}, "deny"),
    ("G1 path traversal", "Write",
     {"file_path": "research/tasks/../state/claims/X.yaml"}, "deny"),
    ("G1 shell redirect into state", "Bash",
     {"command": "echo 'id: X' > research/state/claims/X.yaml"}, "deny"),
    ("G1 sed -i on state", "Bash",
     {"command": "sed -i '' s/a/b/ research/state/claims/CB-AMP-001.yaml"}, "deny"),
    ("G1 cp into state", "Bash",
     {"command": "cp /tmp/x.yaml research/state/claims/x.yaml"}, "deny"),
    ("G1 composed with a benign prefix", "Bash",
     {"command": "ls -la && echo hi > research/state/claims/X.yaml"}, "deny"),
    ("G1 human override", "Edit",
     {"file_path": "research/state/claims/CB-AMP-001.yaml"}, "allow"),  # env below

    # --- G2 push -------------------------------------------------------------
    ("G2 git push", "Bash", {"command": "git push origin main"}, "deny"),
    ("G2 git push composed", "Bash", {"command": "true; git push"}, "deny"),
    ("G2 git -C push", "Bash", {"command": "git -C /repo push --force"}, "deny"),

    # --- G3 destructive git --------------------------------------------------
    ("G3 reset --hard", "Bash", {"command": "git reset --hard origin/main"}, "deny"),
    ("G3 clean -fd", "Bash", {"command": "git clean -fd"}, "deny"),
    ("G3 checkout -f", "Bash", {"command": "git checkout -f main"}, "deny"),
    ("G3 restore", "Bash", {"command": "git restore analysis/"}, "deny"),
    ("G3 rm -rf", "Bash", {"command": "rm -rf results/"}, "deny"),

    # --- G4 HPC / scheduler / remote -----------------------------------------
    # PERMANENT. No gate, no approval and no future HPC access lifts these.
    ("G4 sbatch", "Bash", {"command": "sbatch slurm/run.sh"}, "deny"),
    ("G4 srun composed", "Bash", {"command": "cd slurm && srun -n 4 ./a.out"}, "deny"),
    ("G4 ssh", "Bash", {"command": "ssh ruche 'ls'"}, "deny"),
    ("G4 scontrol", "Bash", {"command": "scontrol show job 12345"}, "deny"),
    ("G4 squeue", "Bash", {"command": "squeue -u utku"}, "deny"),
    ("G4 qsub", "Bash", {"command": "qsub -q normal job.pbs"}, "deny"),
    ("G4 bsub", "Bash", {"command": "bsub < job.lsf"}, "deny"),
    ("G4 condor_submit", "Bash", {"command": "condor_submit job.sub"}, "deny"),
    ("G4 oarsub", "Bash", {"command": "oarsub -l nodes=2 ./run.sh"}, "deny"),
    ("G4 mpirun", "Bash", {"command": "mpirun -np 8 ./solver"}, "deny"),
    ("G4 mpiexec", "Bash", {"command": "mpiexec -n 4 python sim.py"}, "deny"),
    ("G4 sftp", "Bash", {"command": "sftp ruche"}, "deny"),
    ("G4 rsync remote", "Bash", {"command": "rsync -av out/ user@ruche:/scratch/"}, "deny"),
    ("G4 xargs sbatch", "Bash", {"command": "cat jobs.txt | xargs -n1 sbatch"}, "deny"),
    ("G4 sbatch after &&", "Bash", {"command": "make && sbatch run.sh"}, "deny"),
    # Preparing an HPC package is the PERMITTED work. These must NOT be blocked,
    # or the policy becomes unimplementable: agents are required to write,
    # read and validate SLURM scripts for human submission.
    ("OK read a slurm script", "Read", {"file_path": "slurm/run_boundary.sh"}, "allow"),
    ("OK cat a slurm script", "Bash", {"command": "cat slurm/run_boundary.sh"}, "allow"),
    ("OK grep slurm dir", "Bash", {"command": "grep -rn 'ntasks' slurm/"}, "allow"),
    ("OK write a slurm script", "Write",
     {"file_path": "research/tasks/active/TASK-X/proposed/run.sbatch"}, "allow"),
    ("OK shellcheck a slurm script", "Bash",
     {"command": "shellcheck slurm/run_boundary.sh"}, "allow"),
    ("OK local rsync", "Bash", {"command": "rsync -a results/ /tmp/backup/"}, "allow"),
    # Preparation must stay possible. RESOURCE_POLICY section 4 REQUIRES agents
    # to prepare the exact submission command for the researcher, so writing
    # and searching for scheduler names cannot be blocked - only running them.
    ("OK grep for sbatch", "Bash", {"command": "grep -rn sbatch slurm/"}, "allow"),
    ("OK sed a template naming qsub", "Bash",
     {"command": "sed 's/qsub/sbatch/' template.sh > out.sh"}, "allow"),
    ("OK write the submission command for the human", "Bash",
     {"command": 'echo "sbatch --array=1-64 run.sbatch" > READY_FOR_HUMAN_SUBMISSION.txt'}, "allow"),
    ("OK document srun in a memo", "Bash",
     {"command": "printf 'the human runs: srun -n 4 ./a.out\\n' >> notes.md"}, "allow"),
    # ...but indirect execution is still execution.
    ("G4 bash -c wrapping sbatch", "Bash",
     {"command": 'bash -c "sbatch run.sh"'}, "deny"),
    ("G4 eval -c wrapping srun", "Bash",
     {"command": 'sh -c "srun -n 2 ./a.out"'}, "deny"),
    ("G4 nohup sbatch", "Bash", {"command": "nohup sbatch run.sh &"}, "deny"),
    ("G4 env-prefixed sbatch", "Bash",
     {"command": "env FOO=1 sbatch run.sh"}, "deny"),

    # --- G5 manuscripts ------------------------------------------------------
    ("G5 edit main.tex", "Edit",
     {"file_path": "continuousmeasurementslatex/main.tex"}, "deny"),
    ("G5 edit a stray tex", "Write",
     {"file_path": "theory/sec_predictions_revised.tex"}, "deny"),

    # --- G6 known-wrong analysis --------------------------------------------
    ("G6 run anchor_scan", "Bash",
     {"command": ".venv/bin/python3 analysis/anchor_scan.py --L 64"}, "deny"),
    ("G6 import anchor_scan", "Bash",
     {"command": "python -c 'import analysis.anchor_scan'"}, "deny"),
    # Regression: G6 matched any MENTION of the name, so writing the project's
    # own documentation about the known-wrong script was blocked. Discussing a
    # trap is how a trap stays known.
    ("G6 writing docs that name it", "Bash",
     {"command": "cat > notes.md <<'EOF'\nanalysis/anchor_scan.py is known wrong.\nEOF"}, "allow"),
    ("G6 grepping for it", "Bash",
     {"command": "grep -rn anchor_scan research/state/"}, "allow"),

    # --- G7 charter ----------------------------------------------------------
    ("G7 edit charter", "Edit",
     {"file_path": "research/RESEARCH_CHARTER.md"}, "ask"),

    # --- must NOT be blocked (the guard has to stay usable) ------------------
    ("OK read state", "Bash",
     {"command": "cat research/state/claims/CB-AMP-001.yaml"}, "allow"),
    ("OK grep state", "Bash",
     {"command": "grep -rn 'phi' research/state/"}, "allow"),
    ("OK write a task artifact", "Write",
     {"file_path": "research/tasks/active/TASK-X/PROBLEM_MEMO.md"}, "allow"),
    ("OK write a proposal", "Write",
     {"file_path": "research/proposals/TASK-X-redteam/REDTEAM.yaml"}, "allow"),
    ("OK run the validator", "Bash",
     {"command": ".venv/bin/python3 research/tools/validate_state.py"}, "allow"),
    ("OK git status", "Bash", {"command": "git status --porcelain"}, "allow"),
    ("OK git diff", "Bash", {"command": "git diff --stat"}, "allow"),
    ("OK normal rm", "Bash", {"command": "rm /tmp/scratch.txt"}, "allow"),
    ("OK edit HANDOFF", "Edit", {"file_path": "research/HANDOFF.md"}, "allow"),
]


def main() -> int:
    failures = []
    for i, (label, tool, tool_input, expected) in enumerate(CASES):
        env = {"PPSQJ_ALLOW_STATE_WRITE": "1"} if "human override" in label else None
        decision, reason = run(tool, tool_input, env)
        ok = decision == expected
        print(f"{'PASS' if ok else 'FAIL'}  {label:38s} -> {decision}")
        if not ok:
            failures.append(f"{label}: expected {expected}, got {decision} ({reason[:120]})")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    for f in failures:
        print("  FAIL " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
