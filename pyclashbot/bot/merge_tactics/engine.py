import cv2
import numpy as np
import time
import logging
import os
from pyclashbot.bot.merge_tactics import paths
from pyclashbot.bot.merge_tactics.unit_costs import unit_costs
from pyclashbot.bot.merge_tactics.models import GameState, Unit, Action

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self, image_processor):
        self.ip = image_processor
        self._position_cache = {}
        
    def _get_position_name(self, center, grid_cells, bench_cells):
        cache_key = f"{center[0]}_{center[1]}"
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
        x, y = center
        for cell in grid_cells:
            x0, y0, x1, y1 = cell["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._position_cache[cache_key] = cell["name"]
                return cell["name"]
        for cell in bench_cells:
            x0, y0, x1, y1 = cell["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._position_cache[cache_key] = cell["name"]
                return cell["name"]
        return "unknown"
        
    def extract_state(self, screen, elixir_amount, unit_files, unit_imgs, field_imgs, bench_imgs, field_files, bench_files, grid_cells, bench_cells, threshold) -> GameState:
        relevant_templates = []
        relevant_names = []
        
        for i, unit_file in enumerate(unit_files):
            unit_name = os.path.splitext(unit_file)[0]
            unit_cost = unit_costs.get(unit_name, 0)
            relevant_templates.append(unit_imgs[i])
            relevant_names.append(('shop', unit_name))
            
        relevant_templates.extend(field_imgs)
        relevant_templates.extend(bench_imgs)
        relevant_names.extend([('field', os.path.splitext(f)[0]) for f in field_files])
        relevant_names.extend([('bench', os.path.splitext(f)[0]) for f in bench_files])
        
        detections = self.ip.find_units_on_screen(screen, relevant_templates, threshold=threshold, min_dist=30)
        
        state = GameState(
            phase='DEPLOY', 
            elixir=elixir_amount, 
            ruler_hp=100, # Stub for PDF requirement
            modifiers=[], # Stub for PDF requirement
            active_traits=[], # Stub for PDF requirement
            shop_units=[], 
            field_units=[], 
            bench_units=[],
            enemy_units=[] # Stub for PDF Opponent Analyzer
        )
        
        for i, center in detections:
            pos_name = self._get_position_name(center, grid_cells, bench_cells)
            template_type, unit_name = relevant_names[i]
            cost = unit_costs.get(unit_name, 0)
            unit = Unit(name=unit_name, cost=cost, center=center, position=pos_name)
            
            if template_type == 'shop':
                state.shop_units.append(unit)
            elif template_type == 'field' and pos_name.startswith('camp'):
                state.field_units.append(unit)
            elif template_type == 'bench' and pos_name.startswith('banquillo'):
                state.bench_units.append(unit)
                
        return state

class MergeEngine:
    def evaluate(self, state: GameState) -> List[Action]:
        actions = []
        all_units = state.field_units + state.bench_units
        # Find pairs of identical units
        seen = {}
        for u in all_units:
            if u.name in seen:
                # We found a match, generate MERGE action
                target = seen[u.name]
                actions.append(Action(action_type="MERGE", unit_name=u.name, source_pos=u.center, target_pos=target.center))
                return actions # One action at a time
            seen[u.name] = u
        return actions

class PositionEngine:
    def __init__(self):
        # Basic role assignment for frontline (tank/melee) vs backline (ranged)
        self.backline_units = {"archer_queen", "princess", "archers", "musketeer", 
                               "dart_goblin", "magic_archer", "wizard", "ice_wizard", 
                               "electro_wizard", "executioner", "bowler", "bomber",
                               "fire_cracker", "flying_machine"}
                               
    def evaluate(self, state: GameState, grid_cells: list) -> List[Action]:
        actions = []
        if not state.bench_units:
            return actions
            
        # Find empty field cells
        occupied_positions = {u.position for u in state.field_units}
        empty_cells = [cell for cell in grid_cells if cell["name"] not in occupied_positions]
        
        if not empty_cells:
            return actions
            
        bench_unit = state.bench_units[0]
        is_backline = bench_unit.name in self.backline_units
        
        # Sort cells by row number (assuming name is "camp[row,col]")
        # row 0 is frontline, row 3 is backline
        def get_row(cell):
            try:
                # "camp[1,2]" -> "1"
                return int(cell["name"].split("[")[1].split(",")[0])
            except:
                return 0
                
        if is_backline:
            # Prefer higher rows (backline)
            empty_cells.sort(key=get_row, reverse=True)
        else:
            # Prefer lower rows (frontline)
            empty_cells.sort(key=get_row, reverse=False)
            
        target_cell = empty_cells[0]
        
        # Calculate center of target cell
        rect = target_cell["rect"]
        center_x = (rect[0] + rect[2]) // 2
        center_y = (rect[1] + rect[3]) // 2
        
        actions.append(Action(action_type="MOVE", unit_name=bench_unit.name, source_pos=bench_unit.center, target_pos=(center_x, center_y)))
        return actions

class TraitEngine:
    def analyze(self, state: GameState) -> None:
        # Stub: Analyze current traits on the board
        pass

class OpponentAnalyzer:
    def analyze(self, state: GameState) -> None:
        # Stub: Analyze visible enemy units and estimated strength
        pass

class StrategyEngine:
    def __init__(self):
        self.merge_engine = MergeEngine()
        self.position_engine = PositionEngine()
        self.trait_engine = TraitEngine()
        self.opponent_analyzer = OpponentAnalyzer()

    def decide(self, state: GameState, composition: list, grid_cells: list) -> List[Action]:
        actions = []
        
        # 0. Analyze Opponents and Traits (PDF Section 13 & 15)
        self.opponent_analyzer.analyze(state)
        self.trait_engine.analyze(state)
        
        # 1. Merge Engine
        merge_actions = self.merge_engine.evaluate(state)
        if merge_actions:
            return merge_actions
            
        # 2. Position Engine (Move from bench to field)
        move_actions = self.position_engine.evaluate(state, grid_cells)
        if move_actions:
            return move_actions
            
        # 3. Evaluate Shop
        available_compo = [u for u in state.shop_units if u.name in composition and u.cost <= state.elixir]
        if available_compo:
            best_unit = max(available_compo, key=lambda x: x.cost)
            actions.append(Action(action_type="BUY", unit_name=best_unit.name, target_pos=best_unit.center, expected_cost=best_unit.cost))
            return actions # One action at a time
            
        # If no compo, buy cheap stuff if we have a lot of elixir (simulated strategy)
        cheap_units = [u for u in state.shop_units if u.cost <= 5 and u.cost <= state.elixir]
        if cheap_units:
            best_unit = max(cheap_units, key=lambda x: x.cost)
            actions.append(Action(action_type="BUY", unit_name=best_unit.name, target_pos=best_unit.center, expected_cost=best_unit.cost))
            return actions
            
        # 4. Economy & Optimization (Sell non-compo if bench has compo)
        compo_bench = [u for u in state.bench_units if u.name in composition]
        non_compo_field = [u for u in state.field_units if u.name not in composition]
        if compo_bench and non_compo_field:
            cheapest_field = min(non_compo_field, key=lambda x: x.cost)
            actions.append(Action(action_type="SELL", unit_name=cheapest_field.name, target_pos=cheapest_field.center))
            return actions
            
        return actions

class ActionExecutor:
    def __init__(self, adb, image_processor):
        self.adb = adb
        self.ip = image_processor
        
    def execute(self, action: Action, screen_before, vision_engine) -> bool:
        if action.action_type == "BUY":
            logger.info(f"Executing BUY: {action.unit_name}")
            self.adb.tap(*action.target_pos)
            time.sleep(0.5)
            # Verify: check if elixir dropped or shop slot changed
            return True
            
        elif action.action_type == "SELL":
            logger.info(f"Executing SELL: {action.unit_name}")
            self.adb.tap(*action.target_pos)
            time.sleep(0.5)
            screen_mid = self.adb.screenshot()
            sell_btn = self.ip.find_template(screen_mid, paths.SELL_BUTTON)
            if sell_btn:
                self.adb.tap(*sell_btn)
                time.sleep(0.5)
                return True
            return False
            
        elif action.action_type in ("MERGE", "MOVE"):
            logger.info(f"Executing {action.action_type}: {action.unit_name}")
            # Drag and drop from source to target
            self.adb.swipe(
                action.source_pos[0], action.source_pos[1],
                action.target_pos[0], action.target_pos[1]
            )
            time.sleep(0.5)
            return True
            
        return False
