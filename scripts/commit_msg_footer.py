#!/usr/bin/env python3
# =============================================================================
# Commit-Msg Footer — Autonomy & Automation v10 (Tier E4)
# =============================================================================
# Pre-commit `commit-msg` stage hook. Appends a stable automation trailer to
# every commit message so downstream workflows (state-sync.yml, changelog.yml,
# badge-refresh.yml) can detect agent-driven activity deterministically.
#
# Idempotent: never appends the trailer twice. Skips merge / fixup / revert
# commits and any message that already carries an `Automation:` trailer.
# =============================================================================
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

TRAILER_KEY = "Automation"
PLAN_VERSION = "v10"


def should_skip(message: str) -> bool:
    head = message.lstrip().lower()
    if head.startswith(("merge ", "revert ", "fixup!", "squash!", "amend!")):
        return True
    for line in message.splitlines():
        if line.strip().lower().startswith(f"{TRAILER_KEY.lower()}:"):
            return True
    return False


def build_trailer() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{TRAILER_KEY}: plan={PLAN_VERSION} ts={stamp}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 0  # nothing to do
    path = Path(argv[1])
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return 0  # never block the commit on read failure

    if should_skip(original):
        return 0

    trailer = build_trailer()
    sep = "" if original.endswith("\n\n") else ("\n" if original.endswith("\n") else "\n\n")
    updated = f"{original}{sep}{trailer}\n"
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError:
        return 0  # non-fatal
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
