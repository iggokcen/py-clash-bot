from dataclasses import dataclass
from typing import Optional, Tuple, Any

BRIDGE_Y = 320

@dataclass
class PlayIntent:
    card_index: int
    target_pos: Tuple[int, int]
    reason: str = "fallback"

@dataclass
class PlayContext:
    pass

def decide(*args, **kwargs) -> Optional[PlayIntent]:
    return None
