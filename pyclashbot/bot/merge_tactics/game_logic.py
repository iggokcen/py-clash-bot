import cv2
import time
import logging
from pyclashbot.bot.merge_tactics.engine import VisionEngine, StrategyEngine, ActionExecutor

logger = logging.getLogger(__name__)

class GameLogic:
    def __init__(self, adb_controller, image_processor):
        self.adb = adb_controller
        self.image_processor = image_processor
        self.vision = VisionEngine(image_processor)
        self.strategy = StrategyEngine()
        self.executor = ActionExecutor(adb_controller, image_processor)

    def is_bench_empty(self, screen, bench_rect=None, empty_bench_template=None):
        if not bench_rect or not empty_bench_template:
            return False
        x0, y0, x1, y1 = bench_rect
        bench_roi = screen[y0:y1, x0:x1]
        bench_gray = cv2.cvtColor(bench_roi, cv2.COLOR_BGR2GRAY) if bench_roi.ndim == 3 else bench_roi
        empty_template = cv2.imread(empty_bench_template)
        if empty_template is None:
            return False
        tpl_gray = cv2.cvtColor(empty_template, cv2.COLOR_BGR2GRAY) if empty_template.ndim == 3 else empty_template
        if bench_gray.shape != tpl_gray.shape:
            tpl_gray = cv2.resize(tpl_gray, (bench_gray.shape[1], bench_gray.shape[0]))
        res = cv2.matchTemplate(bench_gray, tpl_gray, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)
        return min_val <= 0.13

    def deploy_units(self, screen, composition, elixir_amount, unit_files, unit_imgs, field_imgs, bench_imgs, field_files, bench_files, grid_cells, bench_cells, elixir_rect, threshold=0.8):
        """Refactored to use Vision -> Strategy -> Action Pipeline per Architecture Guidelines"""
        logger.info("[DEPLOY] Phase started using Strategy Pipeline")
        
        # 1. Vision Layer extracts State
        state = self.vision.extract_state(
            screen, elixir_amount, unit_files, unit_imgs, field_imgs, bench_imgs,
            field_files, bench_files, grid_cells, bench_cells, threshold
        )
        logger.info(f"[STATE] Shop:{len(state.shop_units)} Field:{len(state.field_units)} Bench:{len(state.bench_units)} Elixir:{state.elixir}")
        
        # We loop 4 times max to make multiple moves in one phase
        purchases = 0
        for _ in range(4):
            # 2. Strategy Engine decides Action
            actions = self.strategy.decide(state, composition, grid_cells)
            if not actions:
                break
                
            for action in actions:
                # 3. Executor performs Action and verifies
                success = self.executor.execute(action, screen, self.vision)
                if success and action.action_type == "BUY":
                    state.elixir -= action.expected_cost
                    purchases += 1
                    
            # Must re-observe the screen after action
            time.sleep(0.5)
            screen = self.adb.screenshot()
            state = self.vision.extract_state(
                screen, state.elixir, unit_files, unit_imgs, field_imgs, bench_imgs,
                field_files, bench_files, grid_cells, bench_cells, threshold
            )
            
        logger.info(f"[DEPLOY] Completed with {purchases} purchases.")
        return state.elixir
