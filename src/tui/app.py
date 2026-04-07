"""Main TUI application for Gabbie."""

import asyncio
import logging
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from src.tui.screens import DashboardScreen, HistoryScreen, SettingsScreen

log_dir = Path.home() / ".gabbie"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "gabbie.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)


class GabbieTUI(App):
    """Gabbie Voice Gateway TUI Application."""

    CSS = """
Screen {
    background: $background;
}

#dashboard {
    height: 100%;
    padding: 1 2;
}

#status-bar {
    height: 3;
    margin-bottom: 1;
    padding: 0 1;
    background: $surface;
    border: solid $primary;
}

#app-title {
    width: auto;
    content-align: left middle;
    text-style: bold;
    background: $primary;
    color: $text;
    padding: 0 1;
}

#status-indicator {
    width: auto;
    content-align: center middle;
    margin-left: 2;
}

#audio-level-meter {
    width: 14;
    content-align: center middle;
    margin-left: 2;
}

#connection-status {
    width: 1fr;
    content-align: right middle;
    margin-left: 2;
}

#main-grid {
    height: 100%;
    margin-top: 1;
    grid-size: 2;
    grid-rows: 1fr;
    grid-columns: 1fr 2fr;
}

#controls-panel {
    height: 100%;
    padding: 1;
    background: $surface;
    border: solid $primary;
    overflow-y: auto;
}

#log-panel {
    height: 100%;
    padding: 1;
    background: $surface;
    border: solid $primary;
}

#config-summary {
    height: auto;
    margin-top: 1;
    padding: 1;
    background: $panel;
    border: tall $primary;
}

.panel-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
    height: 1;
}

Select {
    width: 100%;
    margin-bottom: 1;
}

Input {
    width: 100%;
    margin-bottom: 1;
}

#button-row {
    width: 100%;
    margin-top: 1;
}

#button-row Button {
    width: 45%;
}

#activity-log {
    height: 1fr;
    background: $panel;
    border: tall $primary;
    padding: 0 1;
}

#history-container {
    padding: 2;
    height: 100%;
}

#history-list {
    height: 100%;
    padding: 1;
    background: $surface;
    border: solid $primary;
}

#settings-container {
    padding: 2;
    height: 100%;
}

#settings-fields {
    height: auto;
    overflow-y: auto;
}

.field-label {
    color: $primary;
    margin-top: 1;
    height: 1;
}

#settings-buttons {
    width: 100%;
    height: auto;
    margin-top: 2;
}

#settings-buttons Button {
    width: 45%;
}

.scrollable {
    overflow-y: auto;
}
"""

    TITLE = "Gabbie Voice Gateway"
    SUB_TITLE = "Voice Interface"

    def on_mount(self) -> None:
        """Install screens on mount."""
        self.install_screen(DashboardScreen(), "dashboard")
        self.install_screen(HistoryScreen(), "history")
        self.install_screen(SettingsScreen(), "settings")
        self.push_screen("dashboard")


def main():
    """Main entry point for TUI."""
    try:
        app = GabbieTUI()
        app.run()
    except Exception as e:
        logger.error(f"TUI error: {e}")
        raise


if __name__ == "__main__":
    main()
