"""Screens for Gabbie TUI."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Select,
    Input,
    Label,
    Switch,
    RichLog,
)
from textual.reactive import reactive

from src.tui.client import DaemonClient
from src.tui.widgets import StatusIndicator


class DashboardScreen(Screen):
    """Main dashboard screen."""

    BINDINGS = [
        ("d", "switch_to_dashboard", "Dashboard"),
        ("h", "show_history", "History"),
        ("s", "show_settings", "Settings"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    connected = reactive(False)
    gateway_state = reactive("idle")

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header(show_clock=True)

        with Container(id="dashboard"):
            # Top status bar
            with Horizontal(id="status-bar"):
                yield Static("Gabbie Voice Gateway", id="app-title")
                yield StatusIndicator(id="status-indicator")
                yield Static("", id="connection-status")

            # Main content area
            with Grid(id="main-grid"):
                # Left panel - Controls
                with Vertical(id="controls-panel"):
                    yield Static("INPUT DEVICE", classes="panel-title")
                    yield Select(
                        options=[],
                        id="input-device-select",
                        allow_blank=True,
                    )

                    yield Static("OUTPUT DEVICE", classes="panel-title")
                    yield Select(
                        options=[],
                        id="output-device-select",
                        allow_blank=True,
                    )

                    yield Static("SERVER URL", classes="panel-title")
                    yield Input(
                        placeholder="http://localhost:8000", id="server-url-input"
                    )

                    with Horizontal(id="button-row"):
                        yield Button("Start", id="start-btn", variant="success")
                        yield Button("Stop", id="stop-btn", variant="error")

                    yield Static("", id="config-summary")

                # Center panel - Activity log
                with Vertical(id="log-panel"):
                    yield Static("ACTIVITY LOG", classes="panel-title")
                    yield RichLog(highlight=True, markup=True, id="activity-log")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "Gabbie Voice Gateway"
        self._connect_to_daemon()

    def _connect_to_daemon(self) -> None:
        """Connect to daemon and load initial data."""
        self.client = DaemonClient()
        if self.client.connect():
            self.connected = True
            self._load_devices()
            self._load_config()
            self._get_status()
        else:
            self.connected = False
            self.query_one("#connection-status", Static).update(
                "[red]Not connected to daemon[/]"
            )

    def _load_devices(self) -> None:
        """Load audio devices from daemon."""
        response = self.client.list_devices()
        if response and response.get("status") == "ok":
            devices = response.get("data", {})

            # Load input devices
            # Select expects (label, value) format
            input_devices = devices.get("input", [])
            input_options = [
                (f"{d['name']} ({d['channels']}ch)", str(d["index"]))
                for d in input_devices
            ]

            input_select = self.query_one("#input-device-select", Select)
            if input_options:
                input_select.set_options(input_options)
                default_input = devices.get("default_input")
                if default_input is not None:
                    matching_option = next(
                        (opt for opt in input_options if opt[1] == str(default_input)),
                        None,
                    )
                    if matching_option:
                        self.call_after_refresh(
                            lambda: setattr(input_select, "value", matching_option[1])
                        )

            # Load output devices
            output_devices = devices.get("output", [])
            output_options = [
                (f"{d['name']} ({d['channels']}ch)", str(d["index"]))
                for d in output_devices
            ]

            output_select = self.query_one("#output-device-select", Select)
            if output_options:
                output_select.set_options(output_options)
                default_output = devices.get("default_output")
                if default_output is not None:
                    matching_option = next(
                        (
                            opt
                            for opt in output_options
                            if opt[1] == str(default_output)
                        ),
                        None,
                    )
                    if matching_option:
                        self.call_after_refresh(
                            lambda: setattr(output_select, "value", matching_option[1])
                        )

    def _load_config(self) -> None:
        """Load configuration from daemon."""
        response = self.client.get_config()
        if response and response.get("status") == "ok":
            config = response.get("data", {})

            # Update server URL
            server_input = self.query_one("#server-url-input", Input)
            server_input.value = config.get("server_url", "")

            # Update config summary
            summary = f"""
[bold]Configuration:[/]
Wake Word: {config.get("wake_word", "N/A")}
STT Model: {config.get("stt_model", "N/A")}
LLM Model: {config.get("llm_model", "N/A")}
TTS Model: {config.get("tts_model", "N/A")}
TTS Voice: {config.get("tts_voice", "N/A")}
            """
            self.query_one("#config-summary", Static).update(summary)

    def _get_status(self) -> None:
        """Get gateway status from daemon."""
        response = self.client.get_status()
        if response and response.get("status") == "ok":
            status = response.get("data", {})
            self.gateway_state = status.get("state", "idle")
            self._update_status_indicator()

    def _update_status_indicator(self) -> None:
        """Update the status indicator widget."""
        indicator = self.query_one("#status-indicator", StatusIndicator)
        indicator.state = self.gateway_state

        # Update connection status
        status_text = f"[green]Connected[/] • State: [bold]{self.gateway_state}[/]"
        self.query_one("#connection-status", Static).update(status_text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start-btn":
            response = self.client.start()
            if response and response.get("status") == "ok":
                self._log(f"[green]{response.get('message')}[/]")
            else:
                message = (
                    response.get("message", "Unknown error")
                    if response
                    else "Unknown error"
                )
                self._log(f"[red]Failed to start: {message}[/]")

        elif event.button.id == "stop-btn":
            response = self.client.stop()
            if response and response.get("status") == "ok":
                self._log(f"[yellow]{response.get('message')}[/]")
            else:
                message = (
                    response.get("message", "Unknown error")
                    if response
                    else "Unknown error"
                )
                self._log(f"[red]Failed to stop: {message}[/]")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle device selection changes."""
        if event.select.id == "input-device-select":
            self.client.update_config(input_device_index=event.value)
            self._log(f"[dim]Input device changed to {event.value}[/]")

        elif event.select.id == "output-device-select":
            self.client.update_config(output_device_index=event.value)
            self._log(f"[dim]Output device changed to {event.value}[/]")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle server URL changes."""
        if event.input.id == "server-url-input":
            self.client.update_config(server_url=event.value)
            self._log(f"[dim]Server URL updated[/]")

    def _log(self, message: str) -> None:
        """Add message to activity log."""
        log = self.query_one("#activity-log", RichLog)
        log.write(message)

    def action_switch_to_dashboard(self) -> None:
        """Switch to dashboard screen."""
        pass  # Already on dashboard

    def action_show_history(self) -> None:
        """Show history screen."""
        self.app.push_screen("history")

    def action_show_settings(self) -> None:
        """Show settings screen."""
        self.app.push_screen("settings")

    def action_refresh(self) -> None:
        """Refresh data from daemon."""
        self._load_devices()
        self._load_config()
        self._get_status()
        self._log("[dim]Refreshed[/]")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class HistoryScreen(Screen):
    """Conversation history screen."""

    BINDINGS = [
        ("d", "switch_to_dashboard", "Dashboard"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the history layout."""
        yield Header(show_clock=True)

        with Vertical(id="history-container"):
            yield Static("CONVERSATION HISTORY", classes="panel-title")

            # Simple text display of history
            with Static(id="history-list", classes="scrollable"):
                yield Static("Loading...", id="history-content")

        yield Footer()

    def on_mount(self) -> None:
        """Load history on mount."""
        self.title = "Conversation History"
        self._load_history()

    def _load_history(self) -> None:
        """Load conversation history from daemon."""
        client = DaemonClient()
        if client.connect():
            response = client.get_status()
            if response and response.get("status") == "ok":
                status = response.get("data", {})
                count = status.get("conversation_count", 0)

                if count > 0:
                    content = f"[green]{count} conversations in memory[/]\n\n[dim](History is shown in the activity log on the dashboard)[/]"
                else:
                    content = "[grey]No conversations yet[/]\n\n[dim]Start the voice gateway and interact to see history[/]"

                self.query_one("#history-content", Static).update(content)

            client.disconnect()

    def action_switch_to_dashboard(self) -> None:
        """Switch to dashboard screen."""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class SettingsScreen(Screen):
    """Settings screen."""

    BINDINGS = [
        ("d", "switch_to_dashboard", "Dashboard"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the settings layout."""
        yield Header(show_clock=True)

        with Vertical(id="settings-container"):
            yield Static("SETTINGS", classes="panel-title")

            with Grid(id="settings-grid"):
                # Wake word settings
                yield Static("WAKE WORD MODEL", classes="panel-title")
                yield Input(
                    value="alexa", placeholder="Wake word model", id="wake-word-input"
                )

                yield Static("DETECTION THRESHOLD", classes="panel-title")
                yield Input(value="0.5", placeholder="0.0 - 1.0", id="threshold-input")

                yield Static("VAD THRESHOLD", classes="panel-title")
                yield Input(value="0.5", placeholder="0.0 - 1.0", id="vad-input")

                # Server settings
                yield Static("STT MODEL", classes="panel-title")
                yield Input(
                    value="Whisper-Tiny", placeholder="STT model", id="stt-input"
                )

                yield Static("LLM MODEL", classes="panel-title")
                yield Input(
                    value="Qwen3.5-122B-A10B-GGUF",
                    placeholder="LLM model",
                    id="llm-input",
                )

                yield Static("TTS MODEL", classes="panel-title")
                yield Input(value="kokoro-v1", placeholder="TTS model", id="tts-input")

                yield Static("TTS VOICE", classes="panel-title")
                yield Input(
                    value="af_bella", placeholder="TTS voice", id="tts-voice-input"
                )

                with Horizontal(id="settings-buttons"):
                    yield Button("Save", id="save-settings-btn", variant="primary")
                    yield Button("Cancel", id="cancel-settings-btn", variant="default")

        yield Footer()

    def on_mount(self) -> None:
        """Load settings on mount."""
        self.title = "Settings"
        self._load_settings()

    def _load_settings(self) -> None:
        """Load current settings from daemon."""
        client = DaemonClient()
        if client.connect():
            response = client.get_config()
            if response and response.get("status") == "ok":
                config = response.get("data", {})

                self.query_one("#wake-word-input", Input).value = config.get(
                    "wake_word", "alexa"
                )
                self.query_one("#threshold-input", Input).value = str(
                    config.get("detection_threshold", 0.5)
                )
                self.query_one("#vad-input", Input).value = str(
                    config.get("vad_threshold", 0.5)
                )
                self.query_one("#stt-input", Input).value = config.get(
                    "stt_model", "Whisper-Tiny"
                )
                self.query_one("#llm-input", Input).value = config.get(
                    "llm_model", "Qwen3.5-122B-A10B-GGUF"
                )
                self.query_one("#tts-input", Input).value = config.get(
                    "tts_model", "kokoro-v1"
                )
                self.query_one("#tts-voice-input", Input).value = config.get(
                    "tts_voice", "af_bella"
                )

            client.disconnect()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-settings-btn":
            self._save_settings()
        elif event.button.id == "cancel-settings-btn":
            self.app.pop_screen()

    def _save_settings(self) -> None:
        """Save settings to daemon."""
        try:
            config = {
                "wake_word": self.query_one("#wake-word-input", Input).value,
                "detection_threshold": float(
                    self.query_one("#threshold-input", Input).value
                ),
                "vad_threshold": float(self.query_one("#vad-input", Input).value),
                "stt_model": self.query_one("#stt-input", Input).value,
                "llm_model": self.query_one("#llm-input", Input).value,
                "tts_model": self.query_one("#tts-input", Input).value,
                "tts_voice": self.query_one("#tts-voice-input", Input).value,
            }

            client = DaemonClient()
            if client.connect():
                response = client.update_config(**config)
                if response and response.get("status") == "ok":
                    self.notify("Settings saved!")
                    self.app.pop_screen()
                else:
                    self.notify("Failed to save settings", severity="error")
                client.disconnect()

        except ValueError:
            self.notify("Invalid threshold values (must be numbers)", severity="error")

    def action_switch_to_dashboard(self) -> None:
        """Switch to dashboard screen."""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
