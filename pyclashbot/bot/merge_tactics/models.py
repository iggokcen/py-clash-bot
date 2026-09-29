from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Unit:
    name: str
    cost: int
    center: Tuple[int, int]
    position: str # 'shop', 'camp[x,y]', or 'banquillo[x]'

@dataclass
class GameState:
    phase: str # 'DEPLOY' or 'BATTLE'
    elixir: int
    ruler_hp: int
    modifiers: List[str]
    active_traits: List[str]
    shop_units: List[Unit]
    field_units: List[Unit]
    bench_units: List[Unit]
    enemy_units: List[Unit]

@dataclass
class Action:
    action_type: str # 'BUY', 'SELL', 'MOVE', 'MERGE'
    unit_name: str
    target_pos: Tuple[int, int]
    source_pos: Optional[Tuple[int, int]] = None
    expected_cost: int = 0
