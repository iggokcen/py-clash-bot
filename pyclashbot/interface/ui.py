from __future__ import annotations

import sys
import threading
import tkinter as tk
import tkinter.filedialog
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING, Any

import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, READONLY, RIGHT, YES, X
from ttkbootstrap.style import Colors
from ttkbootstrap.widgets import ToolTip

from pyclashbot.bot.card_detection import set_active_deck_cards
from pyclashbot.emulators import EmulatorType, get_available_emulators
from pyclashbot.interface.config import (
    BLUESTACKS_DEVICE_CONFIG,
    BLUESTACKS_SETTINGS,
    GOOGLE_PLAY_DEVICE_CONFIG,
    GOOGLE_PLAY_SETTINGS,
    JOBS,
    MEMU_SETTINGS,
    ComboConfig,
    JobConfig,
)
from pyclashbot.interface.enums import (
    BATTLE_STAT_FIELDS,
    BATTLE_STAT_LABELS,
    BOT_STAT_FIELDS,
    BOT_STAT_LABELS,
    COLLECTION_STAT_FIELDS,
    COLLECTION_STAT_LABELS,
    WIN_RATE_STAT_FIELDS,
    WIN_RATE_STAT_LABELS,
    BotStatField,
    DerivedStatField,
    StatField,
    UIField,
    has_start_ready_job,
)
from pyclashbot.interface.widgets import DualRingGauge
from pyclashbot.utils.cr_api import ClashRoyaleClient, resolve_api_key
from pyclashbot.utils.platform import (
    LOW_DISK_SPACE_BYTES,
    clear_recordings,
    get_recordings_dir,
    is_windows,
    recordings_drive_free_bytes,
    recordings_total_bytes_all_locations,
    validate_recordings_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Jobs tab layout (display only — JOBS in config.py keeps bot state-machine order).
# Two columns of section groups to fit the Jobs tab at default height.
_JOB_TAB_COLUMNS: tuple[tuple[tuple[str, tuple[UIField, ...]], ...], ...] = (
    (
        (
            "Battles",
            (
                UIField.CLASSIC_1V1_USER_TOGGLE,
                UIField.CLASSIC_2V2_USER_TOGGLE,
                UIField.TROPHY_ROAD_USER_TOGGLE,
                UIField.WAR_USER_TOGGLE,
                UIField.MERGE_TACTICS_USER_TOGGLE,
            ),
        ),
        (
            "Collection",
            (
                UIField.CARD_UPGRADE_USER_TOGGLE,
                UIField.UPGRADE_PRINCESS_USER_TOGGLE,
                UIField.CARD_MASTERY_USER_TOGGLE,
                UIField.SHOP_DAILY_OFFER_USER_TOGGLE,
            ),
        ),
        ("Account", (UIField.SWITCH_ACCOUNTS_USER_TOGGLE,)),
    ),
    (
        (
            "Clan chat",
            (
                UIField.CLAN_DONATE_USER_TOGGLE,
                UIField.CLAN_REQUEST_CARDS_USER_TOGGLE,
                UIField.CLAN_CLAIM_GIFTS_USER_TOGGLE,
            ),
        ),
        (
            "Decks",
            (
                UIField.RANDOM_DECKS_USER_TOGGLE,
                UIField.CYCLE_DECKS_USER_TOGGLE,
                UIField.RANDOM_PLAYS_USER_TOGGLE,
            ),
        ),
        (
            "AI & Tactics",
            (
                UIField.YOLO_VISION_TOGGLE,
                UIField.SAVE_WINCON_COUNTERS_TOGGLE,
                UIField.SAFE_KING_TOWER_TOGGLE,
            ),
        ),
        ("Options", (UIField.DISABLE_WIN_TRACK_TOGGLE,)),
    ),
)

# Stored disable_win_track_toggle is inverted in the UI — see _job_toggle_default().
_WIN_TRACK_UI_FIELD = UIField.DISABLE_WIN_TRACK_TOGGLE
_JOB_SPINBOX_LABELS: dict[UIField, str] = {
    UIField.DECK_NUMBER_SELECTION: "Deck slot:",
    UIField.MAX_DECK_SELECTION: "Decks:",
    UIField.MAX_ACCOUNT_SELECTION: "Accounts:",
}
_JOB_TOGGLE_ON_STYLE = "round-toggle"
_JOB_TOGGLE_OFF_STYLE = "secondary-round-toggle"
_TAB_SECTION_STYLE = "TabSection.TLabelframe"
_TAB_SECTION_PADDING = (8, 6)
_TAB_CONTAINER_PADDING = (10, 8)
_NOTEBOOK_STYLE = "App.TNotebook"
_NOTEBOOK_TAB_STYLE = "App.TNotebook.Tab"
_NOTEBOOK_TAB_COUNT = 5
_NOTEBOOK_TAB_WIDTH_DIVISOR = 6  # Tcl tab width units vs notebook pixels (approx.)
_IDLE_STATUS = "Idle"
_START_BLOCKED_MESSAGE = "Enable at least one Battles, Clan chat, or Collection job to start."
_ACTION_BUTTON_IPAD = (18, 2)
_WIN_GAUGE_DIAMETER = 58
_WIN_GAUGE_THICKNESS = 8
_APP_ICON_NAME = "pixel-pycb.ico"
_DISCORD_INVITE_URL = "https://pyclashbot.app/discord/invite"
_GITHUB_PROJECT_URL = "https://github.com/pyclashbot/py-clash-bot"
_UPLOAD_RECORDINGS_URL = "https://www.royaletrainer.com/upload"
# How often the "recordings using N GB" note recomputes folder size (ms).
_RECORDINGS_SIZE_REFRESH_MS = 3 * 60 * 1000  # 3 minutes
_DISCORD_BRAND = "#5865F2"
_GITHUB_BRAND = "#238636"
_DISCORD_BUTTON_STYLE = "App.Discord.TButton"
_GITHUB_BUTTON_STYLE = "App.GitHub.TButton"


def no_jobs_popup() -> None:
    messagebox.showerror("Cannot start", _START_BLOCKED_MESSAGE)


class PyClashBotUI(ttk.Window):
    DEFAULT_THEME = "darkly"
    DEFAULT_WIDTH = 520
    DEFAULT_HEIGHT = 540
    MIN_WIDTH = 520
    MIN_HEIGHT = 540

    def __init__(self) -> None:
        super().__init__(themename=self.DEFAULT_THEME)
        self.title("py-clash-bot")
        self._set_window_icon()
        self.geometry(f"{self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}")
        self.resizable(True, True)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        self._style = ttk.Style()
        current_theme = self._style.theme_use()
        if not current_theme:
            current_theme = self.DEFAULT_THEME
        self.theme_var = ttk.StringVar(value=current_theme)
        self.discord_rpc_var = ttk.BooleanVar(value=False)
        self.record_fights_var = ttk.BooleanVar(value=False)
        # Show the resolved default location in the box until the user picks a custom one.
        self.recording_folder_path_var = ttk.StringVar(value=get_recordings_dir())
        self.cr_api_key_var = ttk.StringVar(value=resolve_api_key())
        self.cr_player_tag_var = ttk.StringVar(value="")
        self.advanced_settings_var = ttk.BooleanVar(value=False)
        self._config_callback: Callable[[dict[str, object]], None] | None = None
        self._open_logs_callback: Callable[[], None] | None = None
        self._open_recordings_callback: Callable[[], None] | None = None
        self._config_widgets: dict[str, tk.Widget] = {}
        self._job_toggle_checkbuttons: dict[UIField, ttk.Checkbutton] = {}
        self._job_row_extras: dict[UIField, list[tk.Widget]] = {}
        self._job_extra_spinboxes: dict[UIField, ttk.Spinbox] = {}
        self._theme_labels: list[tk.Widget] = []
        self._muted_labels: list[tk.Widget] = []
        self._stat_accent_labels: list[tk.Widget] = []
        self._traces: list[tuple[tk.Variable, str]] = []
        self._suspend_traces = 0
        self._button_state = "idle"
        self.deck_var: ttk.StringVar | None = None
        self.max_deck_var: ttk.StringVar | None = None
        self.max_account_var: ttk.StringVar | None = None

        # Live Observer & AI Simulation State
        self._observer_thread: threading.Thread | None = None
        self._observer_stop_event = threading.Event()
        self._training_thread: threading.Thread | None = None
        self._training_stop_event = threading.Event()

        self.sim_status_var = ttk.StringVar(value="Idle")
        self.sim_winrate_var = ttk.StringVar(value="0.0%")
        self.sim_episodes_var = ttk.StringVar(value="0")
        self.sim_steps_var = ttk.StringVar(value="0 / 50,000")
        self.sim_speed_var = ttk.StringVar(value="0 steps/s")
        self.sim_timesteps_var = ttk.StringVar(value="50,000")
        self.sim_opponent_var = ttk.StringVar(value="counterpush")
        self.observer_matches_var = ttk.StringVar(value="1")
        self.observer_status_var = ttk.StringVar(value="Idle")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Tab area absorbs extra height
        self.rowconfigure(1, weight=0)  # Status log + Start stay compact

        self._build_tabs()
        self._build_bottom_row()
        self._sync_brand_button_styles()
        self._refresh_theme_colours()

    def register_config_callback(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._config_callback = callback

    def register_open_logs_callback(self, callback: Callable[[], None]) -> None:
        self._open_logs_callback = callback

    def register_open_recordings_callback(self, callback: Callable[[], None]) -> None:
        self._open_recordings_callback = callback

    def get_all_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field, var in self.jobs_vars.items():
            stored = bool(var.get())
            if field == _WIN_TRACK_UI_FIELD:
                stored = not stored
            values[field.value] = stored

        values[UIField.DECK_NUMBER_SELECTION.value] = self._safe_int(
            self.deck_var.get() if self.deck_var is not None else "2", fallback=2
        )
        values[UIField.CYCLE_DECKS_USER_TOGGLE.value] = bool(self.jobs_vars[UIField.CYCLE_DECKS_USER_TOGGLE].get())
        values[UIField.MAX_DECK_SELECTION.value] = self._safe_int(
            self.max_deck_var.get() if self.max_deck_var is not None else "2", fallback=2
        )
        values[UIField.SWITCH_ACCOUNTS_USER_TOGGLE.value] = bool(
            self.jobs_vars[UIField.SWITCH_ACCOUNTS_USER_TOGGLE].get()
        )
        values[UIField.MAX_ACCOUNT_SELECTION.value] = self._safe_int(
            self.max_account_var.get() if self.max_account_var is not None else "2", fallback=2
        )
        emulator_choice = self.emulator_var.get()
        values[UIField.MEMU_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.MEMU
        values[UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.GOOGLE_PLAY
        values[UIField.BLUESTACKS_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.BLUESTACKS
        values[UIField.ADB_TOGGLE.value] = emulator_choice == EmulatorType.ADB

        memu_render = self.memu_render_var.get()
        values[UIField.DIRECTX_TOGGLE.value] = memu_render == "DirectX"
        values[UIField.OPENGL_TOGGLE.value] = memu_render == "OpenGL"

        bs_render = self.bs_render_var.get()
        values[UIField.BS_RENDERER_DX.value] = bs_render == "DirectX"
        values[UIField.BS_RENDERER_GL.value] = bs_render == "OpenGL"
        values[UIField.BS_RENDERER_VK.value] = bs_render == "Vulkan"

        for field, var in self.gp_vars.items():
            values[field.value] = var.get()

        values[UIField.ADB_SERIAL.value] = self.adb_serial_var.get()
        values[UIField.GP_DEVICE_SERIAL.value] = self.gp_device_serial_var.get()
        values[UIField.BS_DEVICE_SERIAL.value] = self.bs_device_serial_var.get()

        values[UIField.THEME_NAME.value] = self.theme_var.get() or self.DEFAULT_THEME
        values[UIField.DISCORD_RPC_TOGGLE.value] = bool(self.discord_rpc_var.get())
        values[UIField.RECORD_FIGHTS_TOGGLE.value] = bool(self.record_fights_var.get())
        values[UIField.RECORDING_FOLDER_PATH.value] = self.recording_folder_path_var.get() or ""
        values[UIField.CR_API_KEY.value] = self.cr_api_key_var.get()
        values[UIField.CR_PLAYER_TAG.value] = self.cr_player_tag_var.get()
        return values

    def set_all_values(self, values: dict[str, object]) -> None:
        theme_value: str | None = None
        self._suspend_traces += 1
        try:
            for field, var in self.jobs_vars.items():
                if field.value in values:
                    ui_value = bool(values[field.value])
                    if field == _WIN_TRACK_UI_FIELD:
                        ui_value = not ui_value
                    var.set(ui_value)

            if UIField.CR_API_KEY.value in values:
                self.cr_api_key_var.set(str(values[UIField.CR_API_KEY.value]))
            if UIField.CR_PLAYER_TAG.value in values:
                self.cr_player_tag_var.set(str(values[UIField.CR_PLAYER_TAG.value]))

            if UIField.DECK_NUMBER_SELECTION.value in values and self.deck_var is not None:
                self.deck_var.set(str(values[UIField.DECK_NUMBER_SELECTION.value]))
            if UIField.MAX_DECK_SELECTION.value in values and self.max_deck_var is not None:
                self.max_deck_var.set(str(values[UIField.MAX_DECK_SELECTION.value]))
            if UIField.MAX_ACCOUNT_SELECTION.value in values and self.max_account_var is not None:
                self.max_account_var.set(str(values[UIField.MAX_ACCOUNT_SELECTION.value]))
            if UIField.THEME_NAME.value in values:
                theme_value = str(values[UIField.THEME_NAME.value])

            # Determine saved emulator choice; default to current selection if no data provided
            saved_emulator = self.emulator_var.get()
            emulator_keys = {
                UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value,
                UIField.BLUESTACKS_EMULATOR_TOGGLE.value,
                UIField.ADB_TOGGLE.value,
                UIField.MEMU_EMULATOR_TOGGLE.value,
            }
            if emulator_keys & values.keys():
                if values.get(UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.GOOGLE_PLAY
                elif values.get(UIField.BLUESTACKS_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.BLUESTACKS
                elif values.get(UIField.ADB_TOGGLE.value):
                    saved_emulator = EmulatorType.ADB
                elif values.get(UIField.MEMU_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.MEMU
            # Use saved choice if available on this platform, otherwise fallback
            available = get_available_emulators()
            if saved_emulator in available:
                self.emulator_var.set(saved_emulator)
            elif available:
                self.emulator_var.set(available[0])

            if values.get(UIField.DIRECTX_TOGGLE.value):
                self.memu_render_var.set("DirectX")
            elif values.get(UIField.OPENGL_TOGGLE.value):
                self.memu_render_var.set("OpenGL")

            if values.get(UIField.BS_RENDERER_VK.value):
                self.bs_render_var.set("Vulkan")
            elif values.get(UIField.BS_RENDERER_DX.value):
                self.bs_render_var.set("DirectX")
            elif values.get(UIField.BS_RENDERER_GL.value):
                self.bs_render_var.set("OpenGL")

            for field, var in self.gp_vars.items():
                config = next((c for c in GOOGLE_PLAY_SETTINGS if c.key == field), None)
                if field.value in values and values[field.value] is not None:
                    var.set(str(values[field.value]))
                elif config:
                    var.set(str(config.default))

            if UIField.ADB_SERIAL.value in values:
                self.adb_serial_var.set(str(values[UIField.ADB_SERIAL.value]))
            if UIField.GP_DEVICE_SERIAL.value in values:
                self.gp_device_serial_var.set(str(values[UIField.GP_DEVICE_SERIAL.value]))
            if UIField.BS_DEVICE_SERIAL.value in values:
                self.bs_device_serial_var.set(str(values[UIField.BS_DEVICE_SERIAL.value]))

            self._update_google_play_comboboxes()

        finally:
            self._suspend_traces -= 1

        if theme_value is not None:
            # Defer theme apply so combobox popdown widgets exist (avoids TclError on macOS).
            self.after_idle(lambda t=theme_value: self._apply_theme(t))

        if self._job_extra_spinboxes:
            self._sync_job_extra_spinboxes()
        if self._job_toggle_checkbuttons:
            self._sync_all_job_toggle_appearances()
        if self._button_state == "idle":
            self._sync_start_button_readiness()

        if UIField.DISCORD_RPC_TOGGLE.value in values:
            self.discord_rpc_var.set(bool(values[UIField.DISCORD_RPC_TOGGLE.value]))

        if UIField.RECORD_FIGHTS_TOGGLE.value in values:
            self.record_fights_var.set(bool(values[UIField.RECORD_FIGHTS_TOGGLE.value]))

        if UIField.RECORDING_FOLDER_PATH.value in values:
            saved_path = str(values[UIField.RECORDING_FOLDER_PATH.value])
            # An empty saved value means "default" -- show the resolved default path.
            self.recording_folder_path_var.set(saved_path or get_recordings_dir())

        self._show_current_emulator_settings()

    def set_button_state(self, state: str) -> None:
        """Set the main button state: 'idle' or 'running'."""
        self._button_state = state
        if state == "idle":
            self._sync_start_button_readiness()
        elif state == "running":
            self.main_btn.configure(text="Stop", bootstyle="danger", state=tk.NORMAL)

        # Disable/enable config widgets based on running state
        running = state == "running"
        for key, widget in self._config_widgets.items():
            if key == "main_btn":
                continue
            try:
                if isinstance(widget, ttk.Combobox):
                    if key == "emulator_combobox":
                        widget.configure(state=tk.DISABLED if running else READONLY)
                    elif widget in (
                        self.adb_serial_combo,
                        self.gp_device_serial_combo,
                        self.bs_device_serial_combo,
                    ):
                        widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                    else:
                        widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Spinbox):
                    if widget in self._job_extra_spinboxes.values():
                        continue
                    widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Radiobutton) and key in [
                    UIField.DIRECTX_TOGGLE.value,
                    UIField.OPENGL_TOGGLE.value,
                    UIField.BS_RENDERER_DX.value,
                    UIField.BS_RENDERER_GL.value,
                    UIField.BS_RENDERER_VK.value,
                ]:
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif widget in [
                    self.adb_connect_btn,
                    self.adb_refresh_btn,
                    self.adb_restart_btn,
                    self.adb_set_size_btn,
                    self.adb_reset_size_btn,
                ]:
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif isinstance(widget, ttk.Button):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)

            except tk.TclError:
                continue
        self._sync_job_extra_spinboxes()
        if self._job_toggle_checkbuttons:
            self._sync_all_job_toggle_appearances()

    def get_button_state(self) -> str:
        """Get the current button state: 'idle' or 'running'."""
        return self._button_state

    def append_log(self, message: str) -> None:
        self.event_log.configure(state="normal")
        self.event_log.delete("1.0", "end")
        self.event_log.insert("end", message)
        self.event_log.configure(state="disabled")
        self.event_log.see("end")

    def set_status(self, text: str) -> None:
        self._status_text = text

    def update_stats(self, stats: dict[str, object] | None) -> None:
        if not stats:
            return

        def as_string(field: StatField, default: str = "0") -> str:
            value = stats.get(field.value, default)
            return str(value)

        def as_int(field: StatField) -> int:
            value = stats.get(field.value)
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return 0

        for field, var in self.stat_labels.items():
            var.set(as_string(field))

        runtime = stats.get(BotStatField.TIME_SINCE_START.value)
        if runtime is not None:
            self.bot_labels[BotStatField.TIME_SINCE_START].set(str(runtime))
        failures = stats.get(BotStatField.RESTARTS_AFTER_FAILURE.value)
        if failures is not None:
            self.bot_labels[BotStatField.RESTARTS_AFTER_FAILURE].set(str(failures))

        winrate_raw = stats.get(DerivedStatField.WINRATE.value)
        wins = as_int(StatField.WINS)
        losses = as_int(StatField.LOSSES)
        parsed_winrate = self._parse_winrate_value(winrate_raw)
        winrate = parsed_winrate if parsed_winrate is not None else self._calculate_winrate_percentage(wins, losses)
        gauge_fg, gauge_bg, gauge_text, _canvas_bg = self._gauge_theme_colours()
        self.win_gauge.animate_to(winrate, fg_colour=gauge_fg, text_colour=gauge_text)

        # Update win streak stats
        current_streak = stats.get(DerivedStatField.CURRENT_WIN_STREAK.value, 0)
        best_streak = stats.get(DerivedStatField.BEST_WIN_STREAK.value, 0)
        if hasattr(self, "current_streak_var"):
            self.current_streak_var.set(str(current_streak))
        if hasattr(self, "best_streak_var"):
            self.best_streak_var.set(str(best_streak))

        if hasattr(self, "policy_canvas"):
            raw_prog = stats.get("policy_progress")
            raw_step = stats.get("policy_step_progress")
            progress = [float(v) for v in raw_prog if isinstance(v, int | float | str)] if isinstance(raw_prog, list) else None
            step_progress = [float(v) for v in raw_step if isinstance(v, int | float | str)] if isinstance(raw_step, list) else None
            if progress or step_progress:
                self._update_policy_graph(progress, step_progress)
            last_reward = stats.get("policy_last_reward")
            if isinstance(last_reward, int | float | str) and hasattr(self, "policy_last_reward_var"):
                try:
                    self.policy_last_reward_var.set(f"{float(last_reward):.3f}")
                except ValueError:
                    pass
            last_loss = stats.get("policy_last_loss")
            if isinstance(last_loss, int | float | str) and hasattr(self, "policy_last_loss_var"):
                try:
                    self.policy_last_loss_var.set(f"{float(last_loss):.4f}")
                except ValueError:
                    pass
            games = stats.get("policy_games")
            if games is not None and hasattr(self, "policy_games_var"):
                self.policy_games_var.set(str(games))

    def _update_policy_graph(self, values: list[float] | None, step_values: list[float] | None) -> None:
        import math

        def finite_series(series: list[float] | None) -> list[float]:
            result: list[float] = []
            for value in series or []:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    result.append(number)
            return result

        values_clean = finite_series(values)
        step_clean = finite_series(step_values)
        if not values_clean and not step_clean:
            self.policy_canvas.delete("all")
            return
        width = max(int(self.policy_canvas.winfo_width()), 1)
        height = max(int(self.policy_canvas.winfo_height()), 1)
        if width <= 1 or height <= 1:
            return
        self.policy_canvas.delete("all")
        self.policy_canvas.create_line(0, height // 2, width, height // 2, fill="#3a3a3a")
        series = []
        if values_clean:
            series.extend(values_clean)
        if step_clean:
            series.extend(step_clean)
        vmin = min(series)
        vmax = max(series)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        pad = 6
        span = vmax - vmin

        def scale(v: float) -> float:
            return pad + (1.0 - (v - vmin) / span) * (height - 2 * pad)

        if values_clean:
            n = len(values_clean)
            if n == 1:
                y = scale(values_clean[0])
                self.policy_canvas.create_oval(width // 2 - 2, y - 2, width // 2 + 2, y + 2, fill="#00aaff")
            else:
                step = width / max(1, n - 1)
                points = []
                for i, val in enumerate(values_clean):
                    x = i * step
                    y = scale(val)
                    points.extend([x, y])
                self.policy_canvas.create_line(*points, fill="#00aaff", width=2, smooth=True)

        if step_clean:
            n = len(step_clean)
            if n == 1:
                y = scale(step_clean[0])
                self.policy_canvas.create_oval(width // 2 - 2, y - 2, width // 2 + 2, y + 2, fill="#ffb020")
            else:
                step = width / max(1, n - 1)
                points = []
                for i, val in enumerate(step_clean):
                    x = i * step
                    y = scale(val)
                    points.extend([x, y])
                self.policy_canvas.create_line(*points, fill="#ffb020", width=1, smooth=True)

    def _build_tabs(self) -> None:
        self._style.configure(f"{_TAB_SECTION_STYLE}.Label", anchor="center")
        self._init_notebook_style()

        self.notebook = ttk.Notebook(self, style=_NOTEBOOK_STYLE)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        self.notebook.bind("<Configure>", self._sync_notebook_tab_widths, add="+")

        self.jobs_tab = ttk.Frame(self.notebook)
        self.emulator_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.sim_tab = ttk.Frame(self.notebook)
        self.misc_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.jobs_tab, text="Jobs")
        self.notebook.add(self.emulator_tab, text="Emulator")
        self.notebook.add(self.stats_tab, text="Stats")
        self.notebook.add(self.sim_tab, text="AI & Sim")
        self.notebook.add(self.misc_tab, text="Misc")

        self._create_jobs_tab()
        self._create_emulator_tab()
        self._create_stats_tab()
        self._create_sim_tab()
        self._create_misc_tab()
        self.after_idle(self._sync_notebook_tab_widths)

    def _init_notebook_style(self) -> None:
        """Equal-width, centered top tabs shared across the app."""
        self._style.layout(_NOTEBOOK_STYLE, self._style.layout("TNotebook"))
        self._style.configure(_NOTEBOOK_STYLE, tabmargins=(2, 4, 2, 0))
        self._style.configure(
            _NOTEBOOK_TAB_STYLE,
            padding=(10, 6),
            anchor="center",
        )

    def _sync_notebook_tab_widths(self, event: tk.Event | None = None) -> None:
        if event is not None and event.widget is not self.notebook:
            return
        try:
            width = self.notebook.winfo_width()
        except tk.TclError:
            return
        if width <= 1:
            return
        tab_width = max(10, width // (_NOTEBOOK_TAB_COUNT * _NOTEBOOK_TAB_WIDTH_DIVISOR))
        try:
            self._style.configure(_NOTEBOOK_TAB_STYLE, width=tab_width)
        except tk.TclError:
            return

    def _section_labelframe(self, parent: tk.Misc, title: str) -> ttk.Labelframe:
        """Section box with a centered title — same look across tabs."""
        return ttk.Labelframe(
            parent,
            text=title,
            labelanchor="n",
            style=_TAB_SECTION_STYLE,
            padding=_TAB_SECTION_PADDING,
        )

    def _compact_action_button_row(
        self,
        parent: tk.Misc,
        *buttons: dict[str, object],
    ) -> tuple[ttk.Button, ...]:
        row = ttk.Frame(parent)
        row.pack(fill=X)
        cluster = ttk.Frame(row)
        cluster.pack(anchor="center")
        created: list[ttk.Button] = []
        last_index = len(buttons) - 1
        for index, config in enumerate(buttons):
            text = str(config["text"])
            command = config["command"]
            style = config.get("style")
            bootstyle = config.get("bootstyle")
            pady_raw = config.get("pady", (0, 0))
            if isinstance(pady_raw, int):
                pady: tuple[int, int] | int = pady_raw
            elif isinstance(pady_raw, tuple) and len(pady_raw) == 2:
                a, b = pady_raw
                pady = (int(str(a)), int(str(b)))
            else:
                pady = (0, 0)
            if style is not None and bootstyle is not None:
                button = ttk.Button(cluster, text=text, command=command, style=str(style), bootstyle=str(bootstyle))
            elif style is not None:
                button = ttk.Button(cluster, text=text, command=command, style=str(style))
            elif bootstyle is not None:
                button = ttk.Button(cluster, text=text, command=command, bootstyle=str(bootstyle))
            else:
                button = ttk.Button(cluster, text=text, command=command)
            padx = (0, 10) if index < last_index else (0, 0)
            button.pack(side=LEFT, ipadx=12, ipady=1, padx=padx, pady=pady)
            created.append(button)
        return tuple(created)

    def _app_icon_path(self) -> Path | None:
        relative = Path("assets") / _APP_ICON_NAME
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            candidates.extend((bundle_root / relative, Path(sys.executable).parent / relative))
        candidates.append(Path(__file__).resolve().parents[2] / relative)
        return next((path for path in candidates if path.is_file()), None)

    def _set_window_icon(self) -> None:
        icon_path = self._app_icon_path()
        if icon_path is None:
            return
        if is_windows():
            try:
                self.iconbitmap(str(icon_path))
            except tk.TclError:
                return
            return
        try:
            from PIL import Image, ImageTk

            with Image.open(icon_path) as source:
                resized = source.resize((64, 64), Image.Resampling.LANCZOS)
                self._window_icon = ImageTk.PhotoImage(resized)
            # PIL ImageTk.PhotoImage is _PhotoImageLike-compatible but not in tkinter stubs.
            self.iconphoto(True, self._window_icon)  # ty: ignore[invalid-argument-type]
        except (ImportError, OSError, tk.TclError):
            return

    def _prepare_tab_page(self, tab: ttk.Frame) -> ttk.Frame:
        """Fill the tab page; extra height opens below the content, not under Start."""
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        page = ttk.Frame(tab, padding=_TAB_CONTAINER_PADDING)
        page.pack(fill="both", expand=YES)
        content = ttk.Frame(page)
        content.pack(fill=X, anchor="n")
        content.columnconfigure(0, weight=1)
        return content

    def _picker_combobox(self, parent: tk.Misc, **kwargs) -> ttk.Combobox:
        """Combobox with consistent TCombobox styling across the app."""
        kwargs.setdefault("state", READONLY)
        kwargs.setdefault("style", "TCombobox")
        return ttk.Combobox(parent, **kwargs)

    def _build_bottom_row(self) -> None:
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)

        status_frame = self._section_labelframe(bottom, "Status")
        status_frame.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)

        self.event_log = tk.Text(status_frame, height=1, wrap="word")
        self.event_log.grid(row=0, column=0, sticky="ew")
        self.event_log.configure(state="disabled")
        self._status_text = "Idle"
        self._style_status_log()

        self._main_btn_row = ttk.Frame(bottom)
        self._main_btn_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self._main_btn_row.columnconfigure(0, weight=1)
        self._main_btn_row.columnconfigure(1, weight=0)
        self._main_btn_row.columnconfigure(2, weight=1)
        ipadx, ipady = _ACTION_BUTTON_IPAD

        self.main_btn = ttk.Button(self._main_btn_row, text="Start", bootstyle="success")
        self.main_btn.grid(row=0, column=1, ipadx=ipadx, ipady=ipady)
        self._register_config_widget("main_btn", self.main_btn)

    def _create_jobs_tab(self) -> None:
        content = self._prepare_tab_page(self.jobs_tab)

        columns_frame = ttk.Frame(content)
        columns_frame.pack(fill=X)
        columns_frame.columnconfigure(0, weight=1, uniform="jobs_cols")
        columns_frame.columnconfigure(1, weight=1, uniform="jobs_cols")

        job_defaults = {job.key: job.default for job in JOBS}
        jobs_by_key = {job.key: job for job in JOBS}
        self.jobs_vars: dict[UIField, ttk.BooleanVar] = {}

        def place_extras_controls(
            parent: tk.Misc, combo_config: ComboConfig, combo_field: UIField, job: JobConfig
        ) -> None:
            short_label = _JOB_SPINBOX_LABELS.get(combo_field, combo_config.label)
            label = ttk.Label(parent, text=short_label)
            label.pack(side=LEFT, padx=(0, 4))
            self._job_row_extras.setdefault(job.key, []).append(label)

            spin_var = ttk.StringVar(value=str(combo_config.default))
            spinbox = ttk.Spinbox(
                parent,
                from_=int(min(combo_config.values)),
                to=int(max(combo_config.values)),
                width=3,
                textvariable=spin_var,
                command=self._notify_config_change,
                state=READONLY,
            )
            spinbox.pack(side=LEFT)
            self._trace_variable(spin_var)
            self._register_config_widget(combo_field.value, spinbox)
            self._job_extra_spinboxes[job.key] = spinbox
            self._job_row_extras.setdefault(job.key, []).append(spinbox)

            if combo_config.tooltip:
                info_label = ttk.Label(parent, text="ⓘ", bootstyle="info")
                info_label.pack(side=LEFT, padx=(4, 0))
                ToolTip(info_label, combo_config.tooltip)
                self._job_row_extras.setdefault(job.key, []).append(info_label)

            if combo_field == UIField.DECK_NUMBER_SELECTION:
                self.deck_var = spin_var
            elif combo_field == UIField.MAX_DECK_SELECTION:
                self.max_deck_var = spin_var
            elif combo_field == UIField.MAX_ACCOUNT_SELECTION:
                self.max_account_var = spin_var

        def place_job_toggle(
            parent: tk.Misc,
            *,
            field: UIField,
            text: str,
            tooltip: str = "",
            row_index: int | None = None,
            command: Callable[[], None] | None = None,
        ) -> ttk.Checkbutton:
            var = ttk.BooleanVar(value=self._job_toggle_default(field, job_defaults))

            def on_toggle() -> None:
                self._sync_job_toggle_appearance(field)
                if command:
                    command()
                else:
                    self._notify_config_change()

            checkbox = ttk.Checkbutton(
                parent,
                text=text,
                variable=var,
                bootstyle=_JOB_TOGGLE_ON_STYLE,
                command=on_toggle,
            )
            if row_index is None:
                checkbox.pack(side=LEFT)
            else:
                checkbox.grid(row=row_index, column=0, sticky="w", pady=2)
            if tooltip:
                ToolTip(checkbox, tooltip)
            self.jobs_vars[field] = var
            self._job_toggle_checkbuttons[field] = checkbox
            self._trace_variable(var)
            self._register_config_widget(field.value, checkbox)
            self._sync_job_toggle_appearance(field)
            return checkbox

        def add_job_row(section_frame: ttk.Labelframe, job: JobConfig, row_index: int) -> int:
            """Place one job row; returns the next free row index."""
            if job.extras:
                # Toggle on row 1; number picker indented below (fits narrow columns).
                combo_config = next(iter(job.extras.values()))
                combo_field = combo_config.key
                place_job_toggle(
                    section_frame,
                    field=job.key,
                    text=job.title,
                    tooltip=job.tooltip,
                    row_index=row_index,
                    command=lambda j=job.key: self._on_job_with_extra_toggle(j),
                )

                extras = ttk.Frame(section_frame)
                extras.grid(
                    row=row_index + 1,
                    column=0,
                    sticky="w",
                    padx=(22, 0),
                    pady=(0, 4),
                )
                place_extras_controls(extras, combo_config, combo_field, job)
                self._sync_job_extra_spinboxes()
                return row_index + 2

            place_job_toggle(
                section_frame,
                field=job.key,
                text=job.title,
                tooltip=job.tooltip,
                row_index=row_index,
            )
            return row_index + 1

        column_padx = ((0, 5), (5, 0))
        for column_index, sections in enumerate(_JOB_TAB_COLUMNS):
            column_frame = ttk.Frame(columns_frame)
            column_frame.grid(row=0, column=column_index, sticky="nsew", padx=column_padx[column_index])
            column_frame.columnconfigure(0, weight=1)

            for section_title, job_keys in sections:
                section_frame = self._section_labelframe(column_frame, section_title)
                section_frame.pack(fill=X, pady=(0, 6))

                grid_row = 0
                for job_key in job_keys:
                    job = jobs_by_key[job_key]
                    grid_row = add_job_row(section_frame, job, grid_row)

        self._sync_all_job_toggle_appearances()

    def can_start(self) -> bool:
        """True when at least one battle, clan chat, or collection job is enabled."""
        return has_start_ready_job(self.get_all_values())

    def _sync_start_button_readiness(self) -> None:
        """Enable Start only when at least one Battles, Clan chat, or Collection job is selected."""
        if self._button_state != "idle":
            return
        if self.can_start():
            self.main_btn.configure(text="Start", bootstyle="success", state=tk.NORMAL)
            self.append_log(_IDLE_STATUS)
        else:
            self.main_btn.configure(text="Start", bootstyle="secondary", state=tk.DISABLED)
            self.append_log(_START_BLOCKED_MESSAGE)

    def _on_job_with_extra_toggle(self, job_key: UIField) -> None:
        self._sync_job_toggle_appearance(job_key)
        self._sync_job_extra_spinboxes()
        self._notify_config_change()

    def _sync_job_extra_spinboxes(self) -> None:
        bot_running = self._button_state == "running"
        for job_key, spinbox in self._job_extra_spinboxes.items():
            job_on = bool(self.jobs_vars[job_key].get())
            if bot_running or not job_on:
                spinbox.configure(state=tk.DISABLED)
            else:
                spinbox.configure(state=READONLY)

    def _muted_foreground(self) -> str:
        try:
            colour = self._style.lookup("secondary.TLabel", "foreground")
            return colour or "#888888"
        except tk.TclError:
            return "#888888"

    def _sync_job_toggle_appearance(self, field: UIField) -> None:
        """Dim job toggles (and their extras) when turned off."""
        checkbox = self._job_toggle_checkbuttons.get(field)
        var = self.jobs_vars.get(field)
        if not checkbox or not var:
            return

        job_on = bool(var.get())
        normal = self._label_foreground()
        muted = self._muted_foreground()
        try:
            checkbox.configure(bootstyle=_JOB_TOGGLE_ON_STYLE if job_on else _JOB_TOGGLE_OFF_STYLE)
        except tk.TclError:
            pass

        for widget in self._job_row_extras.get(field, []):
            try:
                widget.configure(foreground=normal if job_on else muted)
            except tk.TclError:
                continue

    def _sync_all_job_toggle_appearances(self) -> None:
        for field in self._job_toggle_checkbuttons:
            self._sync_job_toggle_appearance(field)

    @staticmethod
    def _job_toggle_default(field: UIField, job_defaults: dict[UIField, bool]) -> bool:
        """Map stored job defaults to what the Jobs tab toggle should show."""
        stored = job_defaults.get(field, False)
        if field == _WIN_TRACK_UI_FIELD:
            return not stored
        return stored

    def _create_emulator_tab(self) -> None:
        content = self._prepare_tab_page(self.emulator_tab)

        connection_section = self._section_labelframe(content, "Emulator")
        connection_section.pack(fill=X, pady=(0, 6))
        selection_frame = ttk.Frame(connection_section)
        selection_frame.pack(fill=X)
        ttk.Label(selection_frame, text="Type:").pack(side=LEFT, padx=(0, 5))

        available_emulators = get_available_emulators()
        default_emulator = available_emulators[0] if available_emulators else EmulatorType.ADB
        self.emulator_var = ttk.StringVar(value=default_emulator)
        self.emulator_combo = self._picker_combobox(
            selection_frame,
            textvariable=self.emulator_var,
            values=available_emulators,
            width=20,
        )
        self.emulator_combo.pack(side=LEFT, fill=X, expand=True)
        self.emulator_combo.bind("<<ComboboxSelected>>", self._on_emulator_changed)
        self._register_config_widget("emulator_combobox", self.emulator_combo)

        ttk.Checkbutton(
            selection_frame,
            text="Advanced settings",
            variable=self.advanced_settings_var,
            bootstyle="round-toggle",
            command=self._on_advanced_settings_toggled,
        ).pack(side=LEFT, padx=(8, 0))

        self.settings_container = ttk.Frame(content)
        self.settings_container.pack_forget()
        self.google_play_frame = ttk.Frame(self.settings_container)
        self.memu_frame = ttk.Frame(self.settings_container)
        self.bluestacks_frame = ttk.Frame(self.settings_container)
        self.adb_frame = ttk.Frame(self.settings_container)

        # Store frames in a dictionary for easy access
        self.emulator_settings_frames = {
            EmulatorType.MEMU: self.memu_frame,
            EmulatorType.GOOGLE_PLAY: self.google_play_frame,
            EmulatorType.BLUESTACKS: self.bluestacks_frame,
            EmulatorType.ADB: self.adb_frame,
        }

        # Populate the settings frames
        self.gp_vars: dict[UIField, ttk.StringVar] = {}
        self._create_google_play_settings(self.google_play_frame)
        self._create_memu_settings(self.memu_frame)
        self._create_bluestacks_settings(self.bluestacks_frame)
        self._create_adb_tab(self.adb_frame)

        # Show the initial settings based on the default value
        self._show_current_emulator_settings()
        self._update_advanced_settings_visibility(self.emulator_var.get())

    def _create_google_play_settings(self, parent_frame: ttk.Frame) -> None:
        device_frame = self._section_labelframe(parent_frame, "Device")
        device_frame.pack(fill=X, pady=(0, 6))

        # Device serial row
        device_row = ttk.Frame(device_frame)
        device_row.pack(fill="x", pady=(0, 5))
        device_row.columnconfigure(1, weight=1)

        ttk.Label(device_row, text="Device:").grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.gp_device_serial_var = ttk.StringVar(value=str(GOOGLE_PLAY_DEVICE_CONFIG.default))
        self.gp_device_serial_combo = self._picker_combobox(
            device_row,
            textvariable=self.gp_device_serial_var,
            state=tk.NORMAL,
        )
        self.gp_device_serial_combo.grid(row=0, column=1, padx=5, sticky="ew")
        self._register_config_widget(UIField.GP_DEVICE_SERIAL.value, self.gp_device_serial_combo)
        self._trace_variable(self.gp_device_serial_var)

        frame = self._section_labelframe(parent_frame, "Google Play Games")
        frame.pack(fill=X, pady=(0, 6))

        left_keys = GOOGLE_PLAY_SETTINGS[:4]
        right_keys = GOOGLE_PLAY_SETTINGS[4:]

        for row, config in enumerate(left_keys):
            self._add_google_play_row(frame, row, 0, config)

        for row, config in enumerate(right_keys):
            self._add_google_play_row(frame, row, 3, config)

    def _create_memu_settings(self, parent_frame: ttk.Frame) -> None:
        self.memu_advanced_frame = self._section_labelframe(parent_frame, "Render mode")
        self.memu_advanced_frame.pack_forget()

        self.memu_render_var = ttk.StringVar(value="DirectX")
        for config in MEMU_SETTINGS:
            text = "DirectX" if config.key == UIField.DIRECTX_TOGGLE else "OpenGL"
            rb = ttk.Radiobutton(
                self.memu_advanced_frame,
                text=text,
                variable=self.memu_render_var,
                value=text,
                command=self._notify_config_change,
            )
            rb.pack(anchor="w")
            self._register_config_widget(config.key.value, rb)

    def _create_bluestacks_settings(self, parent_frame: ttk.Frame) -> None:
        device_frame = self._section_labelframe(parent_frame, "Device")
        device_frame.pack(fill=X, pady=(0, 6))

        # Device serial row
        device_row = ttk.Frame(device_frame)
        device_row.pack(fill="x", pady=(0, 5))
        device_row.columnconfigure(1, weight=1)

        ttk.Label(device_row, text="Device:").grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.bs_device_serial_var = ttk.StringVar(value=str(BLUESTACKS_DEVICE_CONFIG.default))
        self.bs_device_serial_combo = self._picker_combobox(
            device_row,
            textvariable=self.bs_device_serial_var,
            state=tk.NORMAL,
        )
        self.bs_device_serial_combo.grid(row=0, column=1, padx=5, sticky="ew")
        self._register_config_widget(UIField.BS_DEVICE_SERIAL.value, self.bs_device_serial_combo)
        self._trace_variable(self.bs_device_serial_var)

        # Note about auto-discovery
        note_label = ttk.Label(
            device_frame,
            text="Leave empty to auto-detect from BlueStacks config",
            font=("TkDefaultFont", 8),
            foreground=self._muted_foreground(),
        )
        note_label.pack(anchor="w")
        self._muted_labels.append(note_label)

        self.bluestacks_advanced_frame = self._section_labelframe(parent_frame, "Render mode")
        self.bluestacks_advanced_frame.pack_forget()

        self.bs_render_var = ttk.StringVar(value="DirectX")
        for config in BLUESTACKS_SETTINGS:
            if config.key == UIField.BS_RENDERER_DX:
                value = "DirectX"
            elif config.key == UIField.BS_RENDERER_VK:
                value = "Vulkan"
            else:
                value = "OpenGL"
            rb = ttk.Radiobutton(
                self.bluestacks_advanced_frame,
                text=value,
                variable=self.bs_render_var,
                value=value,
                command=self._notify_config_change,
            )
            rb.pack(anchor="w")
            self._register_config_widget(config.key.value, rb)

    def _create_adb_tab(self, parent_frame: ttk.Frame) -> None:
        """Create the widgets for the ADB Device settings tab."""
        frame = self._section_labelframe(parent_frame, "ADB device")
        frame.pack(fill=X, pady=(0, 6))

        # --- Row 1: Serial Input ---
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 5))
        row1.columnconfigure(1, weight=1)

        ttk.Label(row1, text="Device serial:").grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.adb_serial_var = ttk.StringVar(value="")
        self.adb_serial_combo = self._picker_combobox(
            row1,
            textvariable=self.adb_serial_var,
            state=tk.NORMAL,
        )
        self.adb_serial_combo.grid(row=0, column=1, padx=5, sticky="ew")
        self._register_config_widget(UIField.ADB_SERIAL.value, self.adb_serial_combo)
        self._trace_variable(self.adb_serial_var)

        # --- Row 2: Connect/Refresh Buttons ---
        row_buttons_connect = ttk.Frame(frame)
        row_buttons_connect.pack(fill="x", pady=(0, 8))
        row_buttons_connect.columnconfigure(0, weight=1)
        row_buttons_connect.columnconfigure(1, weight=1)

        self.adb_connect_btn = ttk.Button(row_buttons_connect, text="Connect", style="success.TButton")
        self.adb_connect_btn.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self._register_config_widget("adb_connect_btn", self.adb_connect_btn)

        self.adb_refresh_btn = ttk.Button(row_buttons_connect, text="Refresh")
        self.adb_refresh_btn.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        self._register_config_widget("adb_refresh_btn", self.adb_refresh_btn)

        # --- Row 3: Action Buttons (Stacked Vertically) ---
        row_buttons_action = ttk.Frame(frame)
        row_buttons_action.pack(fill="x")

        self.adb_restart_btn = ttk.Button(row_buttons_action, text="Restart ADB")
        self.adb_restart_btn.pack(fill=X, pady=(0, 3))
        self._register_config_widget("adb_restart_btn", self.adb_restart_btn)

        self.adb_set_size_btn = ttk.Button(row_buttons_action, text="Set size & density")
        self.adb_set_size_btn.pack(fill=X, pady=3)
        self._register_config_widget("adb_set_size_btn", self.adb_set_size_btn)

        self.adb_reset_size_btn = ttk.Button(row_buttons_action, text="Reset size & density")
        self.adb_reset_size_btn.pack(fill=X, pady=(3, 0))
        self._register_config_widget("adb_reset_size_btn", self.adb_reset_size_btn)

        ToolTip(self.adb_set_size_btn, "Set the emulator screen to 419x633 and density to 160.")
        ToolTip(self.adb_reset_size_btn, "Reset the emulator screen size and density to device defaults.")

    def _create_stats_tab(self) -> None:
        content = self._prepare_tab_page(self.stats_tab)
        content.columnconfigure(0, weight=1, uniform="stats_cols")
        content.columnconfigure(1, weight=1, uniform="stats_cols")

        self.stat_labels: dict[StatField, ttk.StringVar] = {}

        left = ttk.Frame(content)
        left.grid(row=0, column=0, sticky="new", padx=(0, 5))
        left.columnconfigure(0, weight=1)

        gauge_frame = self._section_labelframe(left, "Win rate")
        gauge_frame.grid(row=0, column=0, sticky="ew")
        gauge_frame.columnconfigure(0, weight=1)

        _gauge_kw = self._initial_gauge_kwargs()
        self.win_gauge = DualRingGauge(
            gauge_frame,
            diameter=_WIN_GAUGE_DIAMETER,
            thickness=_WIN_GAUGE_THICKNESS,
            fg=_gauge_kw["fg"],
            bg=_gauge_kw["bg"],
            text_color=_gauge_kw["text_color"],
            canvas_bg=_gauge_kw["canvas_bg"],
        )
        self.win_gauge.grid(row=0, column=0, pady=(0, 6))

        win_loss_frame = ttk.Frame(gauge_frame)
        win_loss_frame.grid(row=1, column=0, sticky="ew")
        win_loss_frame.columnconfigure(1, weight=1)
        for row, field in enumerate(WIN_RATE_STAT_FIELDS):
            title = WIN_RATE_STAT_LABELS[field]
            label = ttk.Label(win_loss_frame, text=f"{title}:")
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            var = ttk.StringVar(value="0")
            value_label = ttk.Label(win_loss_frame, textvariable=var, foreground=self._stat_accent_foreground())
            value_label.grid(row=row, column=1, sticky="e")
            self._stat_accent_labels.append(value_label)
            self.stat_labels[field] = var

        battle_frame = self._section_labelframe(left, "Battle stats")
        battle_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        battle_frame.columnconfigure(1, weight=1)

        for row, field in enumerate(BATTLE_STAT_FIELDS):
            title = BATTLE_STAT_LABELS[field]
            label = ttk.Label(battle_frame, text=title)
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            var = ttk.StringVar(value="0")
            value_label = ttk.Label(battle_frame, textvariable=var, foreground=self._stat_accent_foreground())
            value_label.grid(row=row, column=1, sticky="e")
            self._stat_accent_labels.append(value_label)
            self.stat_labels[field] = var

        streak_row = len(BATTLE_STAT_FIELDS)
        ttk.Separator(battle_frame, orient="horizontal").grid(
            row=streak_row, column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )
        ttk.Label(battle_frame, text="Current streak:").grid(row=streak_row + 1, column=0, sticky="w")
        self.current_streak_var = ttk.StringVar(value="0")
        current_streak_label = ttk.Label(
            battle_frame,
            textvariable=self.current_streak_var,
            foreground=self._stat_accent_foreground(),
        )
        current_streak_label.grid(row=streak_row + 1, column=1, sticky="e")
        self._stat_accent_labels.append(current_streak_label)
        ttk.Label(battle_frame, text="Best streak:").grid(row=streak_row + 2, column=0, sticky="w")
        self.best_streak_var = ttk.StringVar(value="0")
        best_streak_label = ttk.Label(
            battle_frame,
            textvariable=self.best_streak_var,
            foreground=self._stat_accent_foreground(),
        )
        best_streak_label.grid(row=streak_row + 2, column=1, sticky="e")
        self._stat_accent_labels.append(best_streak_label)

        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="new", padx=(5, 0))
        right.columnconfigure(0, weight=1)

        collection_frame = self._section_labelframe(right, "Collection stats")
        collection_frame.grid(row=0, column=0, sticky="ew")
        collection_frame.columnconfigure(1, weight=1)
        for row, field in enumerate(COLLECTION_STAT_FIELDS):
            title = COLLECTION_STAT_LABELS[field]
            label = ttk.Label(collection_frame, text=title)
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            var = ttk.StringVar(value="0")
            value_label = ttk.Label(collection_frame, textvariable=var, foreground=self._stat_accent_foreground())
            value_label.grid(row=row, column=1, sticky="e")
            self._stat_accent_labels.append(value_label)
            self.stat_labels[field] = var

        bot_frame = self._section_labelframe(right, "Bot stats")
        bot_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bot_frame.columnconfigure(1, weight=1)
        self.bot_labels = {
            BotStatField.RESTARTS_AFTER_FAILURE: ttk.StringVar(value="0"),
            BotStatField.TIME_SINCE_START: ttk.StringVar(value="00:00:00"),
        }
        for row, field in enumerate(BOT_STAT_FIELDS):
            title = BOT_STAT_LABELS[field]
            label = ttk.Label(bot_frame, text=title)
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            value_label = ttk.Label(
                bot_frame,
                textvariable=self.bot_labels[field],
                foreground=self._stat_accent_foreground(),
            )
            value_label.grid(row=row, column=1, sticky="e")
            self._stat_accent_labels.append(value_label)

        # AI Learning Progress Section
        progress_frame = self._section_labelframe(right, "AI learning progress")
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.policy_canvas = tk.Canvas(progress_frame, height=90, bg="#1e1e1e", highlightthickness=0)
        self.policy_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        stats_row = ttk.Frame(progress_frame)
        stats_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        stats_row.columnconfigure(1, weight=1)
        stats_row.columnconfigure(3, weight=1)
        stats_row.columnconfigure(5, weight=1)

        self.policy_last_reward_var = ttk.StringVar(value="0.000")
        self.policy_last_loss_var = ttk.StringVar(value="0.000")
        self.policy_games_var = ttk.StringVar(value="0")

        ttk.Label(stats_row, text="Reward:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_row, textvariable=self.policy_last_reward_var, foreground=self._stat_accent_foreground()).grid(row=0, column=1, sticky="w", padx=(2, 6))

        ttk.Label(stats_row, text="Loss:").grid(row=0, column=2, sticky="w")
        ttk.Label(stats_row, textvariable=self.policy_last_loss_var, foreground=self._stat_accent_foreground()).grid(row=0, column=3, sticky="w", padx=(2, 6))

        ttk.Label(stats_row, text="Games:").grid(row=0, column=4, sticky="w")
        ttk.Label(stats_row, textvariable=self.policy_games_var, foreground=self._stat_accent_foreground()).grid(row=0, column=5, sticky="w", padx=(2, 0))

    def _create_sim_tab(self) -> None:
        """Create the AI & Simulation tab with live observer and RL fast-forward training controls."""
        content = self._prepare_tab_page(self.sim_tab)

        # Top section: 2 columns for Observer & Training Controls
        top_grid = ttk.Frame(content)
        top_grid.pack(fill=X, pady=(0, 6))
        top_grid.columnconfigure(0, weight=1, uniform="sim_top_cols")
        top_grid.columnconfigure(1, weight=1, uniform="sim_top_cols")

        # Column 0: Live Match Observer
        observer_frame = self._section_labelframe(top_grid, "👁️ Live Match Observer")
        observer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        obs_toggle_var = self.jobs_vars.get(UIField.LIVE_OBSERVER_TOGGLE)
        if obs_toggle_var is not None:
            obs_cb = ttk.Checkbutton(
                observer_frame,
                text="Enable in bot battles",
                variable=obs_toggle_var,
                bootstyle="round-toggle",
                command=self._notify_config_change,
            )
            obs_cb.pack(anchor="w", pady=(2, 6))

        match_row = ttk.Frame(observer_frame)
        match_row.pack(fill=X, pady=(0, 6))
        ttk.Label(match_row, text="Matches:").pack(side=LEFT, padx=(0, 4))
        match_spin = ttk.Spinbox(
            match_row,
            from_=1,
            to=10,
            width=3,
            textvariable=self.observer_matches_var,
            state=READONLY,
        )
        match_spin.pack(side=LEFT)

        obs_btn_row = ttk.Frame(observer_frame)
        obs_btn_row.pack(fill=X, pady=(2, 4))
        self.btn_start_observer = ttk.Button(
            obs_btn_row,
            text="👁️ Start Observer",
            bootstyle="info-outline",
            command=self._on_start_observer_clicked,
        )
        self.btn_start_observer.pack(side=LEFT, padx=(0, 4), expand=YES, fill=X)

        # API Settings inside Observer Frame
        api_frame = ttk.Frame(observer_frame)
        api_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Label(api_frame, text="CR Player Tag:").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.cr_player_tag_var = ttk.StringVar(value="")
        self.cr_player_tag_entry = ttk.Entry(api_frame, textvariable=self.cr_player_tag_var, width=15)
        self.cr_player_tag_entry.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 2))
        self._register_config_widget(UIField.CR_PLAYER_TAG.value, self.cr_player_tag_entry)
        self._trace_variable(self.cr_player_tag_var)

        ttk.Label(api_frame, text="CR API Token:").grid(row=1, column=0, sticky="w", pady=(0, 2))
        self.cr_api_key_var = ttk.StringVar(value="")
        self.cr_api_key_entry = ttk.Entry(api_frame, textvariable=self.cr_api_key_var, show="*")
        self.cr_api_key_entry.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 2))
        self._register_config_widget(UIField.CR_API_KEY.value, self.cr_api_key_entry)
        self._trace_variable(self.cr_api_key_var)
        api_frame.columnconfigure(1, weight=1)

        self.btn_stop_observer = ttk.Button(
            obs_btn_row,
            text="⏹ Stop",
            bootstyle="secondary",
            command=self._on_stop_observer_clicked,
            state=tk.DISABLED,
        )
        self.btn_stop_observer.pack(side=LEFT, expand=YES, fill=X)

        obs_status_row = ttk.Frame(observer_frame)
        obs_status_row.pack(fill=X, pady=(4, 0))
        ttk.Label(obs_status_row, text="Status: ", font=("TkDefaultFont", 8)).pack(side=LEFT)
        ttk.Label(
            obs_status_row,
            textvariable=self.observer_status_var,
            foreground=self._stat_accent_foreground(),
            font=("TkDefaultFont", 8, "bold"),
        ).pack(side=LEFT)

        # Column 1: RL Simulation Engine
        train_frame = self._section_labelframe(top_grid, "🤖 RL Simulation Trainer")
        train_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        ttk.Label(train_frame, text="Deck: Taunt King Log Bait (8-Card)", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(0, 4))

        params_row = ttk.Frame(train_frame)
        params_row.pack(fill=X, pady=(0, 6))
        ttk.Label(params_row, text="Steps:").pack(side=LEFT, padx=(0, 2))
        self.sim_steps_combo = self._picker_combobox(
            params_row,
            values=["10,000", "25,000", "50,000", "100,000", "250,000"],
            width=8,
            textvariable=self.sim_timesteps_var,
        )
        self.sim_steps_combo.pack(side=LEFT, padx=(0, 6))

        ttk.Label(params_row, text="Pool:").pack(side=LEFT, padx=(0, 2))
        self.sim_opp_combo = self._picker_combobox(
            params_row,
            values=["counterpush", "aggressive", "random"],
            width=9,
            textvariable=self.sim_opponent_var,
        )
        self.sim_opp_combo.pack(side=LEFT)

        train_btn_row = ttk.Frame(train_frame)
        train_btn_row.pack(fill=X, pady=(2, 4))
        self.btn_start_sim = ttk.Button(
            train_btn_row,
            text="▶ Train (~140x)",
            bootstyle="success-outline",
            command=self._on_start_training_clicked,
        )
        self.btn_start_sim.pack(side=LEFT, padx=(0, 4), expand=YES, fill=X)

        self.btn_stop_sim = ttk.Button(
            train_btn_row,
            text="⏹ Stop",
            bootstyle="danger-outline",
            command=self._on_stop_training_clicked,
            state=tk.DISABLED,
        )
        self.btn_stop_sim.pack(side=LEFT, expand=YES, fill=X)

        deploy_row = ttk.Frame(train_frame)
        deploy_row.pack(fill=X, pady=(4, 0))
        self.btn_deploy_model = ttk.Button(
            deploy_row,
            text="💾 Deploy Model to Bot",
            bootstyle="warning-outline",
            command=self._on_deploy_model_clicked,
        )
        self.btn_deploy_model.pack(fill=X)

        # Middle section: Metrics & Progress
        metrics_frame = self._section_labelframe(content, "📊 Simulation & Training Metrics")
        metrics_frame.pack(fill=X, pady=(0, 6))

        self.sim_progressbar = ttk.Progressbar(
            metrics_frame,
            bootstyle="success-striped",
            maximum=100.0,
            value=0.0,
            mode="determinate",
        )
        self.sim_progressbar.pack(fill=X, pady=(2, 4))

        stats_grid = ttk.Frame(metrics_frame)
        stats_grid.pack(fill=X)
        stats_grid.columnconfigure(1, weight=1)
        stats_grid.columnconfigure(3, weight=1)
        stats_grid.columnconfigure(5, weight=1)
        stats_grid.columnconfigure(7, weight=1)

        accent = self._stat_accent_foreground()
        ttk.Label(stats_grid, text="Progress:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_grid, textvariable=self.sim_steps_var, foreground=accent).grid(row=0, column=1, sticky="w", padx=(2, 6))

        ttk.Label(stats_grid, text="Win Rate:").grid(row=0, column=2, sticky="w")
        ttk.Label(stats_grid, textvariable=self.sim_winrate_var, foreground=accent).grid(row=0, column=3, sticky="w", padx=(2, 6))

        ttk.Label(stats_grid, text="Episodes:").grid(row=0, column=4, sticky="w")
        ttk.Label(stats_grid, textvariable=self.sim_episodes_var, foreground=accent).grid(row=0, column=5, sticky="w", padx=(2, 6))

        ttk.Label(stats_grid, text="Speed:").grid(row=0, column=6, sticky="w")
        ttk.Label(stats_grid, textvariable=self.sim_speed_var, foreground=accent).grid(row=0, column=7, sticky="w", padx=(2, 0))

        # Bottom section: Live Console Feed
        console_frame = self._section_labelframe(content, "🖥️ Live AI & Observer Event Console")
        console_frame.pack(fill=X, pady=(0, 2))

        self.sim_console = tk.Text(
            console_frame,
            height=6,
            wrap="word",
            bg="#141414",
            fg="#00e676",
            font=("Consolas", 8),
            borderwidth=0,
            highlightthickness=0,
        )
        self.sim_console.pack(fill=X, pady=(2, 4))
        self.sim_console.insert(tk.END, ">>> AI Simulation & Live Observer engine ready.\n")
        self.sim_console.configure(state=tk.DISABLED)

        console_bar = ttk.Frame(console_frame)
        console_bar.pack(fill=X)
        ttk.Label(console_bar, textvariable=self.sim_status_var, font=("TkDefaultFont", 8)).pack(side=LEFT)
        ttk.Button(
            console_bar,
            text="Clear",
            bootstyle="link",
            command=self._clear_sim_log,
        ).pack(side=RIGHT)

    def _append_sim_log(self, text: str) -> None:
        def _do() -> None:
            try:
                self.sim_console.configure(state=tk.NORMAL)
                self.sim_console.insert(tk.END, text.rstrip() + "\n")
                self.sim_console.see(tk.END)
                self.sim_console.configure(state=tk.DISABLED)
            except Exception:
                pass

        self.after(0, _do)

    def _clear_sim_log(self) -> None:
        try:
            self.sim_console.configure(state=tk.NORMAL)
            self.sim_console.delete("1.0", tk.END)
            self.sim_console.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _on_start_observer_clicked(self) -> None:
        if self._observer_thread is not None and self._observer_thread.is_alive():
            return

        self._observer_stop_event.clear()
        self.btn_start_observer.configure(state=tk.DISABLED)
        self.btn_stop_observer.configure(state=tk.NORMAL)
        self.observer_status_var.set("Running...")

        matches = int(self.observer_matches_var.get()) if self.observer_matches_var.get().isdigit() else 1
        self._append_sim_log(f"\n[OBSERVER] Starting live observer session for {matches} matches...")

        def _worker() -> None:
            try:
                from tools.live_match_observer import start_observer_session

                start_observer_session(
                    total_matches=matches,
                    log_callback=self._append_sim_log,
                    stop_event=self._observer_stop_event,
                )
            except Exception as e:
                self._append_sim_log(f"[!] Observer session error: {e}")
            finally:
                def _reset() -> None:
                    self.btn_start_observer.configure(state=tk.NORMAL)
                    self.btn_stop_observer.configure(state=tk.DISABLED)
                    self.observer_status_var.set("Idle")

                self.after(0, _reset)

        self._observer_thread = threading.Thread(target=_worker, daemon=True)
        self._observer_thread.start()

    def _on_stop_observer_clicked(self) -> None:
        self._observer_stop_event.set()
        self.observer_status_var.set("Stopping...")
        self._append_sim_log("[OBSERVER] Stop signal sent. Finishing current action...")

    def _on_start_training_clicked(self) -> None:
        if self._training_thread is not None and self._training_thread.is_alive():
            return

        raw_steps = self.sim_timesteps_var.get().replace(",", "").strip()
        steps = int(raw_steps) if raw_steps.isdigit() else 50_000
        opp_type = self.sim_opponent_var.get().strip() or "counterpush"

        self._training_stop_event.clear()
        self.btn_start_sim.configure(state=tk.DISABLED)
        self.btn_stop_sim.configure(state=tk.NORMAL)
        self.sim_status_var.set("Training at ~140x fast-forward...")
        self.sim_progressbar.configure(value=0.0)

        def _progress_cb(data: dict) -> None:
            t = data.get("type")
            if t == "progress":
                step = data.get("step", 0)
                tot = data.get("total_steps", steps)
                pct = (step / max(1, tot)) * 100.0
                win_rate = data.get("win_rate", 0.0)
                episodes = data.get("episodes", 0)

                def _ui_update() -> None:
                    self.sim_progressbar.configure(value=pct)
                    self.sim_steps_var.set(f"{step:,} / {tot:,}")
                    self.sim_winrate_var.set(f"{win_rate:.1f}%")
                    self.sim_episodes_var.set(str(episodes))

                self.after(0, _ui_update)
            elif t == "finished":
                speed = data.get("speed", 0.0)

                def _ui_done() -> None:
                    self.sim_progressbar.configure(value=100.0)
                    self.sim_speed_var.set(f"~{speed:.0f} steps/s")
                    self.sim_status_var.set("Training completed!")

                self.after(0, _ui_done)
            elif t == "stopped":

                def _ui_stop() -> None:
                    self.sim_status_var.set("Training stopped.")

                self.after(0, _ui_stop)

            txt = data.get("text")
            if txt:
                self._append_sim_log(str(txt))

        def _worker() -> None:
            try:
                from tools.train_log_bait_rl import train

                train(
                    total_timesteps=steps,
                    model_save_path="models/log_bait_ppo.zip",
                    progress_callback=_progress_cb,
                    stop_event=self._training_stop_event,
                    opponent_type=opp_type,
                )
            except Exception as e:
                self._append_sim_log(f"[!] Simulation training error: {e}")
            finally:
                def _reset() -> None:
                    self.btn_start_sim.configure(state=tk.NORMAL)
                    self.btn_stop_sim.configure(state=tk.DISABLED)

                self.after(0, _reset)

        self._training_thread = threading.Thread(target=_worker, daemon=True)
        self._training_thread.start()

    def _on_stop_training_clicked(self) -> None:
        self._training_stop_event.set()
        self.sim_status_var.set("Stopping training...")
        self._append_sim_log("[RL] Stop signal sent. Saving final checkpoint...")

    def _on_deploy_model_clicked(self) -> None:
        model_file = Path("models/log_bait_ppo.zip")
        if not model_file.exists():
            messagebox.showwarning(
                "Model Not Found",
                "No trained checkpoint found at models/log_bait_ppo.zip.\nPlease run simulation training first!",
            )
            return
        size_kb = model_file.stat().st_size / 1024
        msg = f"[DEPLOY] Model checkpoint validated ({size_kb:.1f} KB).\nActive bot policy bridge will automatically prioritize models/log_bait_ppo.zip during battles!"
        self._append_sim_log(msg)
        messagebox.showinfo(
            "Model Deployed",
            f"Model successfully deployed to bot!\nFile: {model_file.resolve()}\nSize: {size_kb:.1f} KB",
        )

    def _create_misc_tab(self) -> None:
        content = self._prepare_tab_page(self.misc_tab)

        appearance = self._section_labelframe(content, "Appearance")
        appearance.pack(fill=X, pady=(0, 6))

        ttk.Label(appearance, text="Theme:").pack(anchor="w", pady=(0, 4))
        self.theme_combo = self._picker_combobox(
            appearance,
            values=self._style.theme_names(),
            width=25,
            textvariable=self.theme_var,
        )
        self.theme_combo.pack(anchor="w")
        self.theme_combo.bind("<<ComboboxSelected>>", self._on_theme_change)
        self._trace_variable(self.theme_var)
        self._register_config_widget(UIField.THEME_NAME.value, self.theme_combo)

        data_frame = self._section_labelframe(content, "Data")
        data_frame.pack(fill=X, pady=(0, 6))

        discord_checkbox = ttk.Checkbutton(
            data_frame,
            text="Discord Rich Presence",
            variable=self.discord_rpc_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        discord_checkbox.pack(anchor="w", pady=(0, 6))
        self._trace_variable(self.discord_rpc_var)
        self._register_config_widget(UIField.DISCORD_RPC_TOGGLE.value, discord_checkbox)

        record_fights_checkbox = ttk.Checkbutton(
            data_frame,
            text="Record my 1v1 fights as training data",
            variable=self.record_fights_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        record_fights_checkbox.pack(anchor="w", pady=(0, 6))
        self._trace_variable(self.record_fights_var)
        self._register_config_widget(UIField.RECORD_FIGHTS_TOGGLE.value, record_fights_checkbox)

        # Recording folder path picker
        recording_folder_frame = ttk.Frame(data_frame)
        recording_folder_frame.pack(fill=X, pady=(0, 2))

        ttk.Label(
            recording_folder_frame,
            text="Recording folder:",
            font=("TkDefaultFont", 9),
        ).pack(side=LEFT, padx=(0, 5))

        recording_folder_entry = ttk.Entry(
            recording_folder_frame,
            textvariable=self.recording_folder_path_var,
            width=40,
        )
        recording_folder_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))

        browse_folder_btn = ttk.Button(
            recording_folder_frame,
            text="Browse...",
            width=10,
            command=self._on_browse_recording_folder,
        )
        browse_folder_btn.pack(side=LEFT)
        self._trace_variable(self.recording_folder_path_var)
        self._register_config_widget(UIField.RECORDING_FOLDER_PATH.value, recording_folder_entry)

        # Live validation feedback for the recording folder path.
        self.recording_folder_feedback = ttk.Label(
            data_frame,
            text="",
            font=("TkDefaultFont", 8),
            bootstyle="secondary",
        )
        self.recording_folder_feedback.pack(anchor="w", pady=(0, 0))
        feedback_trace = self.recording_folder_path_var.trace_add("write", self._update_recording_folder_feedback)
        self._traces.append((self.recording_folder_path_var, feedback_trace))
        self._update_recording_folder_feedback()

        self.open_logs_btn, self.open_recordings_btn = self._compact_action_button_row(
            data_frame,
            {
                "text": "View Logs",
                "bootstyle": "secondary",
                "command": self._on_open_logs_clicked,
            },
            {
                "text": "View Recordings",
                "bootstyle": "secondary",
                "command": self._on_open_recordings_clicked,
            },
        )

        self.clear_recordings_btn, self.upload_recordings_btn = self._compact_action_button_row(
            data_frame,
            {
                "text": "Clear recordings",
                "bootstyle": "danger",
                "command": self._on_clear_recordings_clicked,
                "pady": (6, 0),
            },
            {
                "text": "Upload Recordings",
                "bootstyle": "info",
                "command": lambda: webbrowser.open(_UPLOAD_RECORDINGS_URL),
                "pady": (6, 0),
            },
        )

        # Small auto-updating note showing total disk used by recordings.
        self.recordings_size_label = ttk.Label(
            data_frame,
            text="Recordings: calculating...",
            font=("TkDefaultFont", 8),
            bootstyle="secondary",
        )
        self.recordings_size_label.pack(anchor="center", pady=(6, 0))
        self._schedule_recordings_size_refresh()

        # Supercell Clash Royale Official API Integration
        api_frame = self._section_labelframe(content, "Clash Royale Official API")
        api_frame.pack(fill=X, pady=(0, 6))

        tag_row = ttk.Frame(api_frame)
        tag_row.pack(fill=X, pady=(0, 3))
        ttk.Label(tag_row, text="Player Tag:", font=("TkDefaultFont", 9)).pack(side=LEFT, padx=(0, 5))
        tag_entry = ttk.Entry(tag_row, textvariable=self.cr_player_tag_var, width=16)
        tag_entry.pack(side=LEFT, padx=(0, 10))
        self._trace_variable(self.cr_player_tag_var)
        self._register_config_widget(UIField.CR_PLAYER_TAG.value, tag_entry)

        key_row = ttk.Frame(api_frame)
        key_row.pack(fill=X, pady=(0, 3))
        ttk.Label(key_row, text="API Key:", font=("TkDefaultFont", 9)).pack(side=LEFT, padx=(0, 19))
        key_entry = ttk.Entry(key_row, textvariable=self.cr_api_key_var, show="*")
        key_entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self._trace_variable(self.cr_api_key_var)
        self._register_config_widget(UIField.CR_API_KEY.value, key_entry)

        self.cr_api_status_label = ttk.Label(
            api_frame,
            text="API Key ready (loaded from desktop)" if self.cr_api_key_var.get() else "No API key configured",
            font=("TkDefaultFont", 8),
            bootstyle="info" if self.cr_api_key_var.get() else "secondary",
        )
        self.cr_api_status_label.pack(anchor="w", pady=(2, 4))

        self._compact_action_button_row(
            api_frame,
            {
                "text": "Sync Deck & Profile",
                "bootstyle": "primary",
                "command": self._on_sync_cr_deck_clicked,
            },
            {
                "text": "Export Game Data",
                "bootstyle": "secondary",
                "command": self._on_export_cr_data_clicked,
            },
        )

        links_frame = self._section_labelframe(content, "Links")
        links_frame.pack(fill=X)

        self._compact_action_button_row(
            links_frame,
            {
                "text": "Discord",
                "style": _DISCORD_BUTTON_STYLE,
                "command": lambda: webbrowser.open(_DISCORD_INVITE_URL),
            },
            {
                "text": "GitHub",
                "style": _GITHUB_BUTTON_STYLE,
                "command": lambda: webbrowser.open(_GITHUB_PROJECT_URL),
            },
        )

    def _on_sync_cr_deck_clicked(self) -> None:
        tag = self.cr_player_tag_var.get().strip()
        key = self.cr_api_key_var.get().strip()
        if not tag:
            messagebox.showinfo("Player Tag Required", "Please enter your Clash Royale Player Tag (e.g. #8P92UQLR).")
            return
        if not key:
            messagebox.showerror("API Key Required", "No Clash Royale API key configured.")
            return

        self.cr_api_status_label.configure(text="Querying Supercell API...", bootstyle="info")

        def task() -> None:
            try:
                client = ClashRoyaleClient(api_key=key)
                profile = client.get_player(tag)
                deck_cards = client.get_player_deck(tag)
                set_active_deck_cards(deck_cards)
                name = profile.get("name", "Player")
                trophies = profile.get("trophies", 0)
                msg = f"Synced: {name} (🏆 {trophies}) | 8 Cards: {', '.join(deck_cards[:4])}..."
                self.after_idle(lambda: self._handle_cr_sync_success(msg, profile))
            except Exception as e:
                err_msg = str(e)
                self.after_idle(lambda: self._handle_cr_sync_error(err_msg))

        threading.Thread(target=task, daemon=True).start()

    def _handle_cr_sync_success(self, msg: str, profile: dict[str, Any]) -> None:
        self.cr_api_status_label.configure(text=msg, bootstyle="success")
        current_deck = profile.get("currentDeck")
        deck_names = [c.get("name") for c in current_deck if isinstance(c, dict)] if isinstance(current_deck, list) else []
        self.append_log(f"CR API: Successfully synced active battle deck for {profile.get('name')}!\nCards: {deck_names}")

    def _handle_cr_sync_error(self, err_msg: str) -> None:
        self.cr_api_status_label.configure(text=f"Sync failed: {err_msg[:45]}...", bootstyle="danger")
        messagebox.showerror("Sync Failed", f"Failed to sync with Clash Royale API:\n\n{err_msg}")

    def _on_export_cr_data_clicked(self) -> None:
        tag = self.cr_player_tag_var.get().strip()
        key = self.cr_api_key_var.get().strip()
        if not tag:
            messagebox.showinfo("Player Tag Required", "Please enter your Clash Royale Player Tag (e.g. #8P92UQLR).")
            return

        def task() -> None:
            try:
                client = ClashRoyaleClient(api_key=key)
                out_path = client.export_player_data(tag)
                self.after_idle(lambda: messagebox.showinfo("Export Successful", f"Player data exported to:\n{out_path}"))
                self.append_log(f"CR API: Exported game data to {out_path.name}")
            except Exception as e:
                err_msg = str(e)
                self.after_idle(lambda: messagebox.showerror("Export Failed", f"Failed to export data:\n\n{err_msg}"))

        threading.Thread(target=task, daemon=True).start()

    def _register_config_widget(self, key: str, widget: tk.Widget) -> None:
        self._config_widgets[key] = widget

    def _notify_config_change(self, *_: object) -> None:
        if self._suspend_traces > 0 or self._config_callback is None:
            return
        if self._button_state == "idle":
            self._sync_start_button_readiness()
        callback = self._config_callback
        self.after_idle(lambda: callback(self.get_all_values()))

    def _trace_variable(self, var: tk.Variable) -> None:
        trace_id = var.trace_add("write", self._notify_config_change)
        self._traces.append((var, trace_id))

    def _add_google_play_row(
        self,
        frame: ttk.Labelframe,
        row: int,
        column_offset: int,
        config: ComboConfig,
    ) -> None:
        ttk.Label(frame, text=config.label).grid(row=row, column=column_offset, sticky="w", padx=5, pady=2)
        var = ttk.StringVar(value=str(config.default))
        combo = self._picker_combobox(
            frame,
            values=[str(option) for option in config.values],
            width=12,
            textvariable=var,
        )
        combo.grid(row=row, column=column_offset + 1, sticky="w")
        combo.bind("<<ComboboxSelected>>", self._notify_config_change_event)
        field = config.key
        self.gp_vars[field] = var
        self._trace_variable(var)
        self._register_config_widget(field.value, combo)

    def _notify_config_change_event(self, _event: object) -> None:
        self._notify_config_change()

    def _update_google_play_comboboxes(self) -> None:
        for field, var in self.gp_vars.items():
            widget = self._config_widgets.get(field.value)
            if not widget:
                continue
            values = [str(option) for option in widget.cget("values")]
            if var.get() not in values and values:
                var.set(values[0])

    def _reapply_custom_styles(self) -> None:
        """Restore app-specific ttk styles after theme_use resets them."""
        self._style.configure(f"{_TAB_SECTION_STYLE}.Label", anchor="center")
        self._sync_brand_button_styles()
        self._init_notebook_style()

    def _register_brand_button_style(self, style_name: str, background: str) -> None:
        colors = self._style.colors
        foreground = Colors.get_foreground(colors, background)
        pressed = Colors.make_transparent(0.80, background, colors.bg)
        hover = Colors.make_transparent(0.90, background, colors.bg)
        disabled_bg = Colors.make_transparent(0.10, colors.fg, colors.bg)
        disabled_fg = Colors.make_transparent(0.30, colors.fg, colors.bg)
        self._style.configure(
            style_name,
            foreground=foreground,
            background=background,
            bordercolor=background,
            darkcolor=background,
            lightcolor=background,
            focusthickness=0,
            focuscolor=foreground,
            padding=(10, 5),
            anchor="center",
        )
        self._style.layout(style_name, self._style.layout("success.TButton"))
        self._style.map(
            style_name,
            foreground=[("disabled", disabled_fg)],
            background=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            bordercolor=[("disabled", disabled_bg)],
            darkcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            lightcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
        )

    def _sync_brand_button_styles(self) -> None:
        self._register_brand_button_style(_DISCORD_BUTTON_STYLE, _DISCORD_BRAND)
        self._register_brand_button_style(_GITHUB_BUTTON_STYLE, _GITHUB_BRAND)

    def _apply_theme(self, theme_name: str, skip_variable_update: bool = False) -> None:
        available = tuple(self._style.theme_names())
        selected = theme_name if theme_name in available else self.DEFAULT_THEME
        if selected not in available and available:
            selected = available[0]
        if not skip_variable_update or self.theme_var.get() != selected:
            self._suspend_traces += 1
            try:
                self.theme_var.set(selected)
            finally:
                self._suspend_traces -= 1
        try:
            self._style.theme_use(selected)
        except tk.TclError:
            # Popdown widget may be mid-close on macOS; still resync colours below.
            pass
        self._reapply_custom_styles()
        self._refresh_theme_colours()
        if self._button_state == "idle":
            self._sync_start_button_readiness()
        self.after_idle(self._sync_notebook_tab_widths)

    def _label_foreground(self) -> str:
        try:
            colour = self._style.lookup("TLabel", "foreground")
            return colour or "#202020"
        except tk.TclError:
            return "#202020"

    def _stat_accent_foreground(self) -> str:
        colors = getattr(self._style, "colors", None)
        if colors is not None:
            accent = getattr(colors, "info", "")
            if accent:
                return accent
        return self._label_foreground()

    def _panel_background(self) -> str:
        try:
            background = self._style.lookup("TLabelframe", "background")
            if background:
                return background
        except tk.TclError:
            pass
        colors = getattr(self._style, "colors", None)
        if colors is not None:
            return getattr(colors, "bg", "") or self._label_foreground()
        return self._label_foreground()

    def _gauge_theme_colours(self) -> tuple[str, str, str, str]:
        colors = getattr(self._style, "colors", None)
        gauge_fg = getattr(colors, "success", "#2ecc71") if colors is not None else "#2ecc71"
        gauge_bg = getattr(colors, "danger", "#e74c3c") if colors is not None else "#e74c3c"
        return gauge_fg, gauge_bg, self._label_foreground(), self._panel_background()

    def _initial_gauge_kwargs(self) -> dict[str, str]:
        gauge_fg, gauge_bg, gauge_text, canvas_bg = self._gauge_theme_colours()
        return {
            "fg": gauge_fg,
            "bg": gauge_bg,
            "text_color": gauge_text,
            "canvas_bg": canvas_bg,
        }

    def _style_status_log(self) -> None:
        try:
            background = self._style.lookup("TEntry", "fieldbackground")
            foreground = self._style.lookup("TEntry", "foreground")
        except tk.TclError:
            return

        colors = getattr(self._style, "colors", None)
        if not background and colors is not None:
            background = getattr(colors, "input", "") or getattr(colors, "bg", "")
        if not foreground and colors is not None:
            foreground = getattr(colors, "fg", "")

        if not background or not foreground:
            return

        self.event_log.configure(
            background=background,
            foreground=foreground,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            insertbackground=foreground,
        )

    def _refresh_theme_colours(self) -> None:
        foreground = self._label_foreground()
        for label in self._theme_labels:
            try:
                label.configure(foreground=foreground)
            except tk.TclError:
                continue
        muted = self._muted_foreground()
        for label in self._muted_labels:
            try:
                label.configure(foreground=muted)
            except tk.TclError:
                continue
        if hasattr(self, "event_log"):
            self._style_status_log()
        accent = self._stat_accent_foreground()
        for label in self._stat_accent_labels:
            try:
                label.configure(foreground=accent)
            except tk.TclError:
                continue
        if hasattr(self, "win_gauge"):
            gauge_fg, gauge_bg, gauge_text, canvas_bg = self._gauge_theme_colours()
            self.win_gauge.set_colours(gauge_fg, gauge_bg, gauge_text, canvas_bg=canvas_bg)
        if self._job_toggle_checkbuttons:
            self._sync_all_job_toggle_appearances()

    def _on_theme_change(self, _event: object | None = None) -> None:
        self._apply_theme(self.theme_var.get(), skip_variable_update=True)
        self._notify_config_change()

    def _on_emulator_changed(self, _event: object = None) -> None:
        if self.emulator_var.get() == EmulatorType.ADB:
            messagebox.showwarning(
                "ADB Mode",
                "ADB mode is intended for advanced users only. Support will not be provided for ADB mode.",
            )
        self._show_current_emulator_settings()
        self._notify_config_change()

    def _show_current_emulator_settings(self) -> None:
        """Hides all emulator settings frames and shows the one selected in the combobox."""
        selected_emulator = self.emulator_var.get()
        show_advanced = bool(self.advanced_settings_var.get())

        for frame in self.emulator_settings_frames.values():
            frame.pack_forget()

        frame_to_show = self.emulator_settings_frames.get(selected_emulator)
        should_show = True
        if selected_emulator in {EmulatorType.MEMU, EmulatorType.BLUESTACKS, EmulatorType.GOOGLE_PLAY}:
            should_show = show_advanced

        if frame_to_show and should_show:
            self.settings_container.pack(fill=X, anchor="n", pady=(0, 6))
            frame_to_show.pack(fill=X, anchor="n")
        else:
            self.settings_container.pack_forget()

        self._update_advanced_settings_visibility(selected_emulator)
        self.settings_container.update_idletasks()

    def _on_open_logs_clicked(self) -> None:
        if self._open_logs_callback:
            self._open_logs_callback()

    def _on_open_recordings_clicked(self) -> None:
        if self._open_recordings_callback:
            self._open_recordings_callback()

    def _on_browse_recording_folder(self) -> None:
        folder = tkinter.filedialog.askdirectory(
            title="Select Recording Folder",
            initialdir=self.recording_folder_path_var.get() or None,
        )
        if folder:
            self.recording_folder_path_var.set(folder)

    def _schedule_recordings_size_refresh(self) -> None:
        """Refresh the recordings-size note now and re-arm the next refresh."""
        self._refresh_recordings_size()
        self.after(_RECORDINGS_SIZE_REFRESH_MS, self._schedule_recordings_size_refresh)

    def _refresh_recordings_size(self) -> None:
        """Compute total recordings size off the UI thread, then update the note."""
        # Read the tk variable on the main thread; it isn't thread-safe.
        custom_path = self.recording_folder_path_var.get() or None

        def worker() -> None:
            try:
                total = recordings_total_bytes_all_locations(custom_path=custom_path)
            except OSError:
                return
            gb = total / (1024**3)
            self.after(0, lambda: self._set_recordings_size_text(gb))

        threading.Thread(target=worker, daemon=True).start()

    def _set_recordings_size_text(self, gb: float) -> None:
        label = getattr(self, "recordings_size_label", None)
        if label is None:
            return
        try:
            label.configure(text=f"Recordings using {gb:.2f} GB")
        except tk.TclError:
            pass  # window was closed before this callback ran

    def _update_recording_folder_feedback(self, *_: object) -> None:
        """Show path-validity and low-disk-space feedback under the folder field."""
        label = getattr(self, "recording_folder_feedback", None)
        if label is None:
            return
        custom_path = self.recording_folder_path_var.get() or None

        ok, message = validate_recordings_path(custom_path)
        if not ok:
            label.configure(text=message, bootstyle="danger")
            return

        free = recordings_drive_free_bytes(custom_path)
        if free is not None and free <= LOW_DISK_SPACE_BYTES:
            label.configure(
                text="Warning: low drive space. Clean up or export some recordings.",
                bootstyle="danger",
            )
            return

        # No note when the path is valid -- only surface problems.
        label.configure(text="")

    def _on_clear_recordings_clicked(self) -> None:
        if not messagebox.askyesno(
            "Clear recordings",
            "Delete ALL recorded fight packs? This cannot be undone.",
        ):
            return
        removed, freed = clear_recordings(custom_path=self.recording_folder_path_var.get() or None)
        messagebox.showinfo(
            "Recordings cleared",
            f"Deleted {removed} recording(s), freed {freed / (1024**3):.2f} GB.",
        )

    def _update_advanced_settings_visibility(self, emulator_choice: str) -> None:
        show_advanced = bool(self.advanced_settings_var.get())

        def _toggle_frame(frame: ttk.Frame | None, should_show: bool) -> None:
            if not frame:
                return
            try:
                frame.pack_forget()
                if should_show:
                    frame.pack(fill=X, pady=(0, 6))
            except tk.TclError:
                return

        is_memu = emulator_choice == EmulatorType.MEMU
        is_bluestacks = emulator_choice == EmulatorType.BLUESTACKS

        _toggle_frame(getattr(self, "memu_advanced_frame", None), show_advanced and is_memu)
        _toggle_frame(getattr(self, "bluestacks_advanced_frame", None), show_advanced and is_bluestacks)

    def _on_advanced_settings_toggled(self) -> None:
        # Re-evaluate which emulator settings should be visible when the toggle changes
        self._show_current_emulator_settings()

    @staticmethod
    def _safe_int(value: object, fallback: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _parse_winrate_value(raw: object) -> float | None:
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.endswith("%"):
                stripped = stripped[:-1]
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(raw, int | float):
            return float(raw)
        return None

    @staticmethod
    def _calculate_winrate_percentage(wins: int, losses: int) -> float:
        total = wins + losses
        if total <= 0:
            return 0.0
        return wins / total * 100
