from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Add KataCR path if present so torch unpickling never fails
_kata_path = Path(__file__).resolve().parents[2] / "KataCR"
if _kata_path.exists() and str(_kata_path) not in sys.path:
    sys.path.append(str(_kata_path))

from pyclashbot.bot.board import Unit
from pyclashbot.bot.card_knowledge import facts_for

_MODEL: YOLO | None = None
_MODEL_CHECKED = False
_YOLO_ENABLED = True


def set_yolo_enabled(enabled: bool) -> None:
    global _YOLO_ENABLED
    _YOLO_ENABLED = enabled


def _find_model_path() -> Path | None:
    candidates = [
        Path.cwd() / "clash_royale_detector1.pt",
        Path.cwd() / "clash_royale_yolo.pt",
        Path(__file__).resolve().parent / "clash_royale_detector1.pt",
        Path(__file__).resolve().parent / "clash_royale_yolo.pt",
        Path(__file__).resolve().parent.parent.parent / "clash_royale_detector1.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def is_yolo_available() -> bool:
    return _YOLO_ENABLED and load_model() is not None


def load_model() -> YOLO | None:
    global _MODEL, _MODEL_CHECKED
    if not _MODEL_CHECKED:
        _MODEL_CHECKED = True
        model_path = _find_model_path()
        if model_path is not None:
            try:
                _MODEL = YOLO(str(model_path))
            except Exception:
                _MODEL = None
        else:
            _MODEL = None
    return _MODEL


_IGNORE_CLASSES = {
    "king-tower",
    "queen-tower",
    "cannoneer-tower",
    "dagger-duchess-tower",
    "dagger-duchess-tower-bar",
    "tower-bar",
    "king-tower-bar",
    "clock",
    "emote",
    "elixir",
    "evolution-symbol",
    "padding_belong",
    "bar-level",
    "text",
    "selected",
}


def detect_units_yolo(bgr: np.ndarray, conf: float = 0.22) -> tuple[Unit, ...]:
    """Detect all units on arena using YOLOv8, returning both ally and enemy units."""
    model = load_model()
    if model is None:
        return ()

    if bgr.shape[0] < 510 or bgr.shape[1] < 385:
        return ()

    results = model.predict(source=bgr, conf=conf, verbose=False)

    units: list[Unit] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls_id]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Only consider units inside the playable arena
            if cy < 70 or cy > 505 or class_name in _IGNORE_CLASSES:
                continue

            if class_name == "bar":
                # Determine ally vs enemy by health bar color (red=enemy, blue=ally)
                sub = bgr[max(0, y1) : min(bgr.shape[0], y2), max(0, x1) : min(bgr.shape[1], x2)]
                is_enemy = bool(sub[:, :, 2].mean() > sub[:, :, 0].mean()) if sub.size > 0 else True
                card_id = None
                is_air = False
            else:
                is_enemy = not class_name.startswith("ally_")
                card_id = (
                    class_name.removeprefix("enemy_")
                    .removeprefix("ally_")
                    .lower()
                    .strip()
                    .replace("-", "_")
                )
                facts = facts_for(card_id)
                is_air = facts.is_air

            # Spatial deduplication
            if any(abs(u.x - cx) < 25 and abs(u.y - cy) < 25 for u in units):
                continue

            units.append(
                Unit(
                    x=cx,
                    y=cy,
                    is_air=is_air,
                    enemy=is_enemy,
                    card_id=card_id,
                )
            )

    return tuple(units)


def detect_enemy_units_yolo(bgr: np.ndarray) -> tuple[Unit, ...]:
    """Detect enemy units using YOLOv8."""
    return tuple(u for u in detect_units_yolo(bgr) if u.enemy)
