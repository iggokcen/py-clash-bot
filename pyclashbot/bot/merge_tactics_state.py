import time
import json
import logging
from pyclashbot.bot.merge_tactics.game_logic import GameLogic
from pyclashbot.bot.merge_tactics.image_processing import ImageProcessor
from pyclashbot.bot.merge_tactics.composition_manager import CompositionManager
from pyclashbot.bot.merge_tactics import paths

logger = logging.getLogger(__name__)

class EmulatorWrapper:
    def __init__(self, emulator):
        self.emulator = emulator
        
    def screenshot(self):
        # The pyclashbot emulator returns BGR numpy array
        return self.emulator.screenshot()
        
    def tap(self, x, y):
        self.emulator.click(x, y)

def merge_tactics_state(emulator, logger_param):
    """
    Main state for Merge Tactics game mode.
    Will enter the game loop and play until it detects the end of the match.
    Returns: Next state string or boolean.
    """
    logger_param.change_status("Starting Merge Tactics Mode")
    
    adb_wrapper = EmulatorWrapper(emulator)
    image_processor = ImageProcessor()
    game_logic = GameLogic(adb_wrapper, image_processor)
    comp_manager = CompositionManager(paths.COMPOSITION_FILE)
    
    composition = comp_manager.load_composition()
    if not composition:
        logger_param.change_status("Merge Tactics: No composition found, skipping.")
        return True
        
    unit_files, unit_imgs, _ = paths.load_unit_images()
    field_files, field_imgs = paths.load_field_images()
    bench_files, bench_imgs = paths.load_bench_images()
    
    with open(paths.GRID_FILE, "r") as f:
        grid_data = json.load(f)
    grid_cells = [item for item in grid_data if item["name"].startswith("camp")]
    bench_cells = [item for item in grid_data if item["name"].startswith("banquillo")]
    
    with open(paths.ELIXIR_FILE, "r") as f:
        elixir_data = json.load(f)
    elixir_rect = next(item["rect"] for item in elixir_data if item["name"]=="elixir")
    
    last_elixir_value = 0
    match_active = True
    
    start_time = time.time()
    idle_start_time = time.time()
    
    while match_active:
        current_time = time.time()
        if current_time - start_time > 300: # 5 minutes timeout per match
            logger_param.change_status("Merge Tactics timeout.")
            break
            
        if current_time - idle_start_time > 30:
            logger_param.change_status("Merge Tactics: Stuck for 30 seconds. Triggering restart.")
            return False # Will trigger handle_state_failure
            
        if current_time - idle_start_time > 15:
            logger_param.change_status("Merge Tactics: Idle for 15s. Tapping to dismiss pop-up.")
            adb_wrapper.tap(10, 300) # Tap empty corner
            idle_start_time = current_time # Reset to avoid spamming
            time.sleep(1.0)
            
        screen = adb_wrapper.screenshot()
        action_taken = False
        
        # Check for reward
        reward_pos = image_processor.find_template(screen, paths.REWARD, paths.THRESHOLD)
        if reward_pos:
            logger_param.change_status("Merge Tactics: Collecting reward")
            adb_wrapper.tap(*reward_pos)
            action_taken = True
            time.sleep(1.0)
            
        # Check for battle initialization
        elif not action_taken and (init_pos := image_processor.find_template(screen, paths.INIT_BATTLE, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Starting battle")
            adb_wrapper.tap(*init_pos)
            action_taken = True
            time.sleep(1.0)
            
        # Check for play again
        elif not action_taken and (play_again := image_processor.find_template(screen, paths.PLAY_AGAIN_BATTLE, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Playing again")
            adb_wrapper.tap(*play_again)
            action_taken = True
            time.sleep(1.0)
            
        # Check for back button / quit to exit state
        elif not action_taken and (back := image_processor.find_template(screen, paths.BACK, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Exiting match")
            adb_wrapper.tap(*back)
            break
            
        elif not action_taken and (quit_pos := image_processor.find_template(screen, paths.QUIT, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Quitting match")
            adb_wrapper.tap(*quit_pos)
            break
            
        # Check for deploy phase
        elif not action_taken and (deploy := image_processor.find_template(screen, paths.PHASE_DEPLOY, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Deploying")
            elixir_value = image_processor.read_elixir(screen, elixir_rect)
            if elixir_value is not None:
                last_elixir_value = elixir_value
                
            if last_elixir_value > 0:
                # Pass logger_param down if possible, but game_logic is already using standard logger.
                # game_logic output is handled later in game_logic.py edit.
                last_elixir_value = game_logic.deploy_units(
                    screen, composition, last_elixir_value,
                    unit_files, unit_imgs, field_imgs, bench_imgs,
                    field_files, bench_files, grid_cells, bench_cells,
                    elixir_rect, paths.THRESHOLD
                )
            action_taken = True
            time.sleep(0.5)
            
        # Check for battle phase
        elif not action_taken and (battle_phase := image_processor.find_template(screen, paths.PHASE_BATTLE, paths.THRESHOLD)):
            logger_param.change_status("Merge Tactics: Battle ongoing")
            action_taken = True
            time.sleep(2.0)
            
        if action_taken:
            idle_start_time = time.time()
        else:
            time.sleep(1.0)
        
    return True
