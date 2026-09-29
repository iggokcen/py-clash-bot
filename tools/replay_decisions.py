"""Replay a recorded pack through the strategy engine and report what it would do.

This is the measurement loop the engine was designed for. No emulator, no game:
the pack's frames are fed to ``read_board`` one at a time and the resulting
decisions are scored, so placement rules can be tuned on a machine that has no
account running.

Run it with::

    uv run python -m tools.replay_decisions <pack-slug>

With no argument it picks the newest pack that has frames. It prints, per play,
the elixir read, the rule that fired and the reason -- the same strings the bot
writes to its log during a live fight, so the two can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy
from PIL import Image

from pyclashbot.bot.board import read_board
from pyclashbot.bot.card_knowledge import facts_for
from pyclashbot.bot.decide import PlayContext, decide

# The deck the calibration pack was played with. Swap per player.
DECK = (
    "ice_wizard",
    "hero_knight",
    "tombstone",
    "fireball",
    "royal_delivery",
    "mega_minion",
    "little_prince",
    "hog",
)


@dataclass
class Tick:
    index: int
    elixir: int
    card_id: str
    x: int
    y: int
    rule: str
    reason: str
    score: float


def recordings_root() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise SystemExit("APPDATA is not set; cannot locate the recordings directory")
    return os.path.join(appdata, "py-clash-bot", "recordings")


def find_pack(slug: str | None) -> str:
    root = recordings_root()
    if not os.path.isdir(root):
        raise SystemExit(f"no recordings directory at {root}")
    candidates = []
    for name in sorted(os.listdir(root)):
        frames = os.path.join(root, name, "frames")
        if os.path.isdir(frames) and any(n.endswith(".png") for n in os.listdir(frames)):
            candidates.append((os.path.getmtime(os.path.join(root, name)), name, frames))
    if not candidates:
        raise SystemExit(f"no pack with frames under {root}")
    if slug:
        for _, name, frames in candidates:
            if name == slug:
                return frames
        raise SystemExit(f"pack {slug!r} has no frames")
    candidates.sort()
    return candidates[-1][2]


def load_bgr(path: str) -> numpy.ndarray:
    return numpy.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def hand_for(elixir: int, cycle: int) -> list[str]:
    """Rotate the deck like the game does, then keep only affordable cards."""
    hand = list(DECK[cycle % len(DECK) :]) + list(DECK[: cycle % len(DECK)])
    return [c for c in hand if facts_for(c).elixir <= elixir]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?", help="pack slug; defaults to the newest")
    parser.add_argument("--stride", type=int, default=1, help="evaluate every Nth frame")
    args = parser.parse_args(argv)

    frames_dir = find_pack(args.slug)
    names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))
    print(f"pack frames: {frames_dir}")
    print(f"frames: {len(names)}  stride: {args.stride}")

    history: list[str] = []
    ticks: list[Tick] = []
    elixirs: list[int] = []
    rules: Counter[str] = Counter()
    cards: Counter[str] = Counter()

    for i, name in enumerate(names):
        if i % args.stride:
            continue
        elapsed = i / 3.0
        board = read_board(load_bgr(os.path.join(frames_dir, name)), elapsed=elapsed)
        elixirs.append(board.elixir)
        hand = hand_for(board.elixir, cycle=i)
        if not hand:
            continue
        intent = decide(board, hand, PlayContext(recent_plays=tuple(history[-6:])))
        if intent is None:
            continue
        history.append(intent.card_id)
        rules[intent.rule] += 1
        cards[intent.card_id] += 1
        ticks.append(
            Tick(i, board.elixir, intent.card_id, intent.x, intent.y, intent.rule, intent.reason, intent.score)
        )

    if not ticks:
        print("engine produced no plays")
        return 1

    print()
    print(f"plays decided: {len(ticks)}")
    print(f"elixir read: min={min(elixirs)} max={max(elixirs)} mean={sum(elixirs) / len(elixirs):.2f}")
    print(f"rules fired: {dict(rules)}")
    print(f"card spread: {dict(cards)}")

    repeats = sum(1 for a, b in pairwise(ticks) if a.card_id == b.card_id)
    print(f"consecutive same-card plays: {repeats}/{len(ticks) - 1}")

    spells = [t for t in ticks if facts_for(t.card_id).is_spell]
    on_tower = [t for t in spells if abs(t.y - 134) <= 8]
    print(f"spell plays: {len(spells)}  landing within 8px of an enemy tower row: {len(on_tower)}")

    print()
    print("first 20 decisions:")
    for t in ticks[:20]:
        print(f"  f{t.index:04d} elixir={t.elixir:2d} {t.card_id:<14} ({t.x:3d},{t.y:3d}) {t.rule:<18} {t.reason}")

    out = os.path.join(os.path.dirname(frames_dir), "decisions.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t in ticks:
            f.write(
                json.dumps(
                    {
                        "frame_index": t.index,
                        "elixir": t.elixir,
                        "card_id": t.card_id,
                        "x": t.x,
                        "y": t.y,
                        "rule": t.rule,
                        "reason": t.reason,
                        "score": t.score,
                    }
                )
                + "\n"
            )
    print()
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
