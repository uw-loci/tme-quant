#!/usr/bin/env python3
"""
Called by the PostCompact and Stop hooks to keep session-summary.md current.

PostCompact: always update (compact is a meaningful context milestone).
Stop:        only update when the repo HEAD is not yet reflected in the file,
             i.e. new commits exist since the last summary write.

The script calls `claude -p` to regenerate the summary with AI-quality prose.
A lock file prevents recursion: the `claude -p` sub-session fires its own Stop
hook, which sees the lock and exits immediately.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── recursion / overlap guard ─────────────────────────────────────────────────
LOCK = Path(tempfile.gettempdir()) / "ctfire_update_summary.lock"
if LOCK.exists():
    sys.exit(0)

try:
    LOCK.touch()

    # ── hook event context ────────────────────────────────────────────────────
    try:
        hook_data = json.load(sys.stdin)
    except Exception:
        hook_data = {}
    event = hook_data.get("hook_event_name", "Stop")

    # ── repo root ─────────────────────────────────────────────────────────────
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(0)
    repo = Path(r.stdout.strip())
    summary = repo / ".claude" / "session-summary.md"
    if not summary.exists():
        sys.exit(0)

    # ── decide whether to act ─────────────────────────────────────────────────
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if not head:
        sys.exit(0)

    if event == "Stop":
        # Only proceed when commits exist that aren't reflected in the summary yet
        if head in summary.read_text(encoding="utf-8"):
            sys.exit(0)
    # PostCompact → always proceed (context just rolled over, summary may be stale)

    # ── require `claude` to be on PATH ────────────────────────────────────────
    if not shutil.which("claude"):
        print("[hook] claude not found in PATH — skipping summary update",
              file=sys.stderr)
        sys.exit(0)

    # ── invoke claude headless to rewrite the summary ─────────────────────────
    prompt = (
        "Update .claude/session-summary.md to reflect the current project state. "
        "First run 'git log --oneline -10' and 'git status --short' to gather facts. "
        "Keep the existing markdown structure and section headings. "
        "Update: the date in the header to today, the commits table, "
        "the branch-state section, and the possible-next-steps section. "
        "Rewrite narrative sections ('What was done', 'Key fixes') only when "
        "new commits add genuinely new information. "
        "Do not add new sections. Write the result back to the file."
    )
    subprocess.run(
        ["claude", "-p", prompt],
        cwd=str(repo),
        check=False,
        timeout=180,
    )
    print(f"[hook] session-summary.md updated (event={event}, HEAD={head})",
          file=sys.stderr)

finally:
    LOCK.unlink(missing_ok=True)
