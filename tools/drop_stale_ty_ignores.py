"""Remove stale `# ty: ignore[unresolved-attribute]` comments that ty flags as unused.

ty 0.0.49 resolves winreg, os.startfile, ctypes.windll and
subprocess.DETACHED_PROCESS on Windows, so the suppressions these files carry are
no longer needed -- and an unused suppression is itself a `unused-ignore-comment`
warning, which is what failed `make lint`.

A comment that documented *why* the call is Windows-only is kept as a plain
comment; only the suppression marker goes. Written with binary-safe I/O because
these files contain em dashes and box-drawing glyphs that a naive round-trip
would mangle.

Usage:  uv run python tools\\drop_stale_ty_ignores.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys

# "...  # ty: ignore[rule]  # why"  ->  "...  # why"
#
# The trailing \r? matters: these files are CRLF on disk, so splitting on "\n"
# leaves a \r on every line and a plain `$` anchor silently fails to match the
# multi-line-call sites. Only the last two markers were dropping before this.
_MARKER = re.compile(r"[ \t]*#\s*ty:\s*ignore\[[a-z-]+\][ \t]*(#[^\r\n]*)?\r?$")

TARGETS = (
    "pyclashbot/emulators/bluestacks.py",
    "pyclashbot/emulators/google_play.py",
    "pyclashbot/emulators/memu.py",
    "pyclashbot/utils/machine_info.py",
    "pyclashbot/utils/open_folder.py",
)


def clean(path: str) -> tuple[int, str]:
    with open(path, encoding="utf-8", newline="") as handle:
        original = handle.read()
    out_lines: list[str] = []
    dropped = 0
    for line in original.split("\n"):
        match = _MARKER.search(line)
        if match is None:
            out_lines.append(line)
            continue
        dropped += 1
        trailing = (match.group(1) or "").strip()
        code = line[: match.start()].rstrip()
        out_lines.append(f"{code}  {trailing}" if trailing else code)
    return dropped, "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    args = parser.parse_args()

    total = 0
    for path in TARGETS:
        dropped, updated = clean(path)
        total += dropped
        status = "would drop" if args.check else "dropped"
        print(f"{path}: {status} {dropped}")
        if dropped and not args.check:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
    print(f"total suppressions removed: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
