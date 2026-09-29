from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    name: str | None = None
    bot_id: str | None = None
    side: str | None = None


@dataclass
class DetectionSummary:
    total: int
    top: int
    mid: int
    bottom: int
    left: int
    right: int
    max_conf: float


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "field_detector.onnx"
DEFAULT_INPUT_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
SIDE_BUFFER_FRAC = float(os.getenv("PYCLASHBOT_SIDE_BUFFER_FRAC", "0.06"))
SIDE_MEMORY_TTL = float(os.getenv("PYCLASHBOT_SIDE_MEMORY_TTL", "2.5"))
SIDE_SWITCH_COUNT = int(os.getenv("PYCLASHBOT_SIDE_SWITCH_COUNT", "2"))
_SIDE_MEMORY: dict[tuple[int, int, int], dict[str, float | str | int]] = {}

ROBOFLOW_CLASSES = [
    'ally_archer', 'ally_arrow', 'ally_baby_dragon', 'ally_balloon', 'ally_bandit',
    'ally_barbarian', 'ally_bat', 'ally_battleram', 'ally_bomber', 'ally_bombtower',
    'ally_bowler', 'ally_brawler', 'ally_cursedhog', 'ally_dark_prince',
    'ally_electro_wizard', 'ally_elite_barbarian', 'ally_elixirpump', 'ally_executioner',
    'ally_fire_spirit', 'ally_fireball', 'ally_firecracker', 'ally_fisherman',
    'ally_giant', 'ally_giant_bomb', 'ally_giant_skeleton', 'ally_goblin',
    'ally_goblin_barrel', 'ally_goblin_cage', 'ally_goblin_hut', 'ally_goblindrill',
    'ally_golem_mini', 'ally_graveyard', 'ally_guard', 'ally_hog_rider',
    'ally_hungry_dragon', 'ally_hunter', 'ally_ice_golem', 'ally_ice_spirit',
    'ally_ice_wizard', 'ally_infernodragon', 'ally_knight', 'ally_knight_evo',
    'ally_log', 'ally_lumberjack', 'ally_magic_archer', 'ally_mega_knight',
    'ally_mega_minion', 'ally_mightyminer', 'ally_mini_pekka', 'ally_minion',
    'ally_mortar', 'ally_motherwitch', 'ally_musketeer', 'ally_pekka', 'ally_poison',
    'ally_prince', 'ally_princess', 'ally_rage', 'ally_rocket', 'ally_royale_giant',
    'ally_royalghost', 'ally_skeleton', 'ally_skeletonking', 'ally_spear_goblin',
    'ally_tesla', 'ally_tesla_hidden', 'ally_tombstone', 'ally_tornado',
    'ally_valkyrie', 'ally_wallbreaker', 'ally_witch', 'ally_wizard', 'ally_xbow',
    'ally_zappy', 'enemy_archer', 'enemy_arrow', 'enemy_baby_dragon', 'enemy_bandit',
    'enemy_barbarian', 'enemy_barbarian_evo', 'enemy_bat', 'enemy_bomb_tower',
    'enemy_bomber', 'enemy_brawler', 'enemy_dark_prince', 'enemy_dart_goblin',
    'enemy_electro_wizard', 'enemy_electrogiant', 'enemy_elixirgolem',
    'enemy_executioner', 'enemy_fire_spirit', 'enemy_fireball', 'enemy_firecracker',
    'enemy_furnace', 'enemy_giant', 'enemy_giant_bomb', 'enemy_giant_skeleton',
    'enemy_goblin', 'enemy_goblin_barrel', 'enemy_goblin_cage', 'enemy_goblin_hut',
    'enemy_goblinbarrel', 'enemy_goldenknight', 'enemy_golem', 'enemy_golem_mini',
    'enemy_guard', 'enemy_hog', 'enemy_hog_rider', 'enemy_hunter', 'enemy_ice_golem',
    'enemy_inferno_tower', 'enemy_infernodragon', 'enemy_knight', 'enemy_log',
    'enemy_lumberjack', 'enemy_magic_archer', 'enemy_mega_knight', 'enemy_mega_minion',
    'enemy_miner', 'enemy_mini_pekka', 'enemy_minion', 'enemy_musketeer',
    'enemy_night_witch', 'enemy_pekka', 'enemy_prince', 'enemy_princess', 'enemy_rage',
    'enemy_ramrider', 'enemy_rascalboy', 'enemy_rascalgirl', 'enemy_rocket',
    'enemy_royal_ghost', 'enemy_skeleton', 'enemy_skeleton_barrel',
    'enemy_skeleton_evo', 'enemy_sparky', 'enemy_spear_goblin', 'enemy_tesla',
    'enemy_tesla_hidden', 'enemy_tombstone', 'enemy_valkyrie', 'enemy_witch',
    'enemy_wizard', 'enemy_xbow',
]


def _det_key(det: Detection, cell: int) -> tuple[int, int, int]:
    cx = int((det.x1 + det.x2) / 2)
    cy = int((det.y1 + det.y2) / 2)
    return (int(det.cls_id), int(cx // cell), int(cy // cell))


class OnnxTroopDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        class_names: list[str] | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        conf_threshold: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Detector model not found: {self.model_path}")
        self.class_names = class_names or ROBOFLOW_CLASSES
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return blob, (w, h)

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        blob, (orig_w, orig_h) = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        if not outputs:
            return []
        pred = np.asarray(outputs[0])
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes = pred[:, :4]
        scores = pred[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]

        keep = confs >= self.conf_threshold
        boxes = boxes[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]
        if boxes.size == 0:
            return []

        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        xyxy[:, [0, 2]] *= scale_x
        xyxy[:, [1, 3]] *= scale_y

        boxes_nms = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
        scores_nms = confs.tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0]
        indices = np.array(indices).reshape(-1)

        detections: list[Detection] = []
        for i in indices:
            x1, y1, x2, y2 = xyxy[i]
            name, bot_id, side = _resolve_class(self.class_names, int(cls_ids[i]))
            x1 = int(max(0, min(orig_w, x1)))
            y1 = int(max(0, min(orig_h, y1)))
            x2 = int(max(0, min(orig_w, x2)))
            y2 = int(max(0, min(orig_h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(confs[i]),
                    cls_id=int(cls_ids[i]),
                    name=name,
                    bot_id=bot_id,
                    side=side,
                )
            )
        return detections


def summarize_detections(detections: list[Detection], image_shape: tuple[int, int, int]) -> DetectionSummary:
    if not detections:
        return DetectionSummary(0, 0, 0, 0, 0, 0, 0.0)
    h, w = image_shape[:2]
    mid_x = w / 2.0
    top_cut = h * 0.35
    mid_cut = h * 0.6

    top = mid = bottom = left = right = 0
    max_conf = 0.0
    for det in detections:
        cx = (det.x1 + det.x2) / 2.0
        cy = (det.y1 + det.y2) / 2.0
        if cy < top_cut:
            top += 1
        elif cy < mid_cut:
            mid += 1
        else:
            bottom += 1
        if cx < mid_x:
            left += 1
        else:
            right += 1
        max_conf = max(det.conf, max_conf)
    return DetectionSummary(
        total=len(detections),
        top=top,
        mid=mid,
        bottom=bottom,
        left=left,
        right=right,
        max_conf=max_conf,
    )


def classify_detection_side(image: np.ndarray, det: Detection) -> str:
    """Classify ally vs enemy based on HP bar color above bounding box."""
    if image is None or image.size == 0:
        return "unknown"
    h, w = image.shape[:2]
    y1 = max(0, det.y1 - 6)
    y2 = max(0, det.y1 - 1)
    x1 = max(0, det.x1 + 6)
    x2 = min(w, det.x2 - 6)
    if y2 <= y1 or x2 <= x1:
        return "unknown"
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    mean = roi.reshape(-1, 3).mean(axis=0)
    b, g, r = float(mean[0]), float(mean[1]), float(mean[2])
    red_score = r - max(g, b)
    blue_score = b - max(r, g)
    if r > 120 and red_score > 25:
        return "enemy"
    if b > 90 and blue_score > 20:
        return "ally"
    return "unknown"


def split_detections_by_side(
    image: np.ndarray,
    detections: list[Detection],
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Split detections into (enemy, ally, unknown)."""
    if image is None or image.size == 0:
        return [], [], detections
    h, w = image.shape[:2]
    mid_y = h * 0.5
    buffer = h * SIDE_BUFFER_FRAC
    now = time.time()
    cell = max(24, int(min(h, w) * 0.05))
    enemy: list[Detection] = []
    ally: list[Detection] = []
    unknown: list[Detection] = []
    for det in detections:
        side = det.side
        candidate = side
        if candidate is None:
            candidate = classify_detection_side(image, det)
        if candidate == "unknown":
            cy = (det.y1 + det.y2) / 2.0
            if cy < mid_y - buffer:
                candidate = "enemy"
            elif cy > mid_y + buffer:
                candidate = "ally"
            else:
                candidate = "unknown"

        key = _det_key(det, cell)
        state = _SIDE_MEMORY.get(key)
        final_side = candidate

        if candidate in ("enemy", "ally"):
            if state is None or state.get("side") == candidate:
                final_side = candidate
                _SIDE_MEMORY[key] = {
                    "side": candidate,
                    "ts": now,
                    "pending_side": "",
                    "pending_count": 0,
                }
            else:
                pending_side = str(state.get("pending_side", "") or "")
                pending_count = int(state.get("pending_count", 0))
                if pending_side == candidate:
                    pending_count += 1
                else:
                    pending_side = candidate
                    pending_count = 1
                if pending_count >= SIDE_SWITCH_COUNT:
                    final_side = candidate
                    _SIDE_MEMORY[key] = {
                        "side": candidate,
                        "ts": now,
                        "pending_side": "",
                        "pending_count": 0,
                    }
                else:
                    final_side = str(state.get("side", candidate))
                    _SIDE_MEMORY[key] = {
                        "side": final_side,
                        "ts": now,
                        "pending_side": pending_side,
                        "pending_count": pending_count,
                    }
        else:
            if state and now - float(state.get("ts", 0.0)) <= SIDE_MEMORY_TTL:
                final_side = str(state.get("side", "unknown"))
            else:
                final_side = "unknown"
            if state:
                state["ts"] = now
                _SIDE_MEMORY[key] = state

        if final_side == "enemy":
            enemy.append(det)
        elif final_side == "ally":
            ally.append(det)
        else:
            unknown.append(det)

    if _SIDE_MEMORY:
        cutoff = now - (SIDE_MEMORY_TTL * 2.0)
        stale_keys = [k for k, v in _SIDE_MEMORY.items() if float(v.get("ts", 0.0)) < cutoff]
        for k in stale_keys:
            _SIDE_MEMORY.pop(k, None)
    return enemy, ally, unknown


def load_detector_from_env() -> OnnxTroopDetector | None:
    path = os.getenv("PYCLASHBOT_FIELD_DETECTOR")
    target = Path(path) if path else DEFAULT_MODEL_PATH
    if not target.exists():
        return None
    try:
        return OnnxTroopDetector(model_path=target)
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned.startswith("ally_"):
        cleaned = cleaned[len("ally_") :]
    elif cleaned.startswith("enemy_"):
        cleaned = cleaned[len("enemy_") :]
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    special = {
        "the_log": "log",
        "x_bow": "xbow",
        "mini_p_e_k_k_a": "mini_pekka",
        "p_e_k_k_a": "pekka",
        "giant_snowball": "snowball",
        "firecracker": "fire_cracker",
        "royal_hog": "royal_hogs",
        "spear_goblins": "spear_goblins",
    }
    return special.get(cleaned, cleaned)


def _split_team_label(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    lowered = name.lower()
    if lowered.startswith("ally_"):
        return "ally", name[len("ally_") :]
    if lowered.startswith("enemy_"):
        return "enemy", name[len("enemy_") :]
    return None, name


def _resolve_class(class_names: list[str], cls_id: int) -> tuple[str | None, str | None, str | None]:
    if not class_names or cls_id < 0 or cls_id >= len(class_names):
        return None, None, None
    name = class_names[cls_id]
    side, base = _split_team_label(name)
    return name, _normalize_name(base or name), side
