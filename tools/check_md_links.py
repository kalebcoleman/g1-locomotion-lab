#!/usr/bin/env python3
"""Check relative links and image sources in the tracked Markdown files.

Only local, relative targets are checked (http(s), mailto and bare anchors are skipped). The target is resolved
against the directory of the file that contains it; a missing target is reported as `path:line: target`.

Usage: python3 tools/check_md_links.py
Exit code: 0 if every link resolves, 1 otherwise. No third-party dependencies.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse

LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)|<img[^>]+src="([^"]+)"')
SKIP = ("http://", "https://", "mailto:", "#")


def tracked_markdown(root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        print("warning: not a git checkout, scanning the working tree", file=sys.stderr)
        return [os.path.relpath(os.path.join(d, f), root)
                for d, _, files in os.walk(root) for f in files if f.endswith(".md") and ".git" not in d]
    return [p for p in out.stdout.split("\0") if p]


def main() -> int:
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or os.getcwd()
    files = tracked_markdown(root)
    broken = []
    for rel in files:
        path = os.path.join(root, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            broken.append((rel, 0, f"cannot read: {exc}"))
            continue
        for match in LINK.finditer(text):
            target = match.group(1) or match.group(2)
            if not target or target.startswith(SKIP):
                continue
            local = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not local:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), local))
            if not (os.path.exists(resolved) or os.path.exists(resolved + ".md")):
                broken.append((rel, text[:match.start()].count("\n") + 1, target))
    print(f"{len(files)} markdown files checked, {len(broken)} broken relative link(s)")
    for rel, line, target in broken:
        print(f"{rel}:{line}: {target}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
