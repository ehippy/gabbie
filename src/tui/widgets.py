"""Custom widgets for Gabbie TUI."""

from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Horizontal


class StatusIndicator(Widget):
    """Animated status indicator widget."""

    state: reactive[str] = reactive("idle")

    def __init__(self, state: str = "idle", **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def watch_state(self, old_state: str, new_state: str) -> None:
        """Update display when state changes."""
        self.refresh()

    def render(self) -> str:
        """Render the status indicator."""
        colors = {
            "idle": "grey",
            "listening": "green",
            "detected": "yellow",
            "recording": "yellow",
            "processing": "cyan",
            "speaking": "blue",
            "error": "red",
        }

        color = colors.get(self.state, "grey")

        state_labels = {
            "idle": "IDLE",
            "listening": "LISTENING",
            "detected": "DETECTED",
            "recording": "RECORDING",
            "processing": "PROCESSING",
            "speaking": "SPEAKING",
            "error": "ERROR",
        }

        label = state_labels.get(self.state, self.state.upper())

        if self.state == "listening":
            # Animated listening effect
            return f"[{color}]●[/] [{color}]{label}[/]"
        elif self.state == "processing":
            return f"[{color}]◷[/] [{color}]{label}[/]"
        elif self.state == "speaking":
            return f"[{color}]🔊[/] [{color}]{label}[/]"
        elif self.state == "error":
            return f"[{color}]⚠[/] [{color}]{label}[/]"
        else:
            return f"[{color}]○[/] [{color}]{label}[/]"


class AudioLevelMeter(Widget):
    """Simple audio level visualization."""

    level: reactive[float] = reactive(0.0)

    def __init__(self, level: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.level = level

    def watch_level(self, old_level: float, new_level: float) -> None:
        """Refresh display when level changes."""
        self.refresh()

    def render(self) -> str:
        """Render the audio level meter."""
        blocks = [" ", "░", "▒", "▓", "█"]
        num_blocks = 10
        filled = int(self.level * num_blocks)

        meter = ""
        for i in range(num_blocks):
            if i < filled:
                meter += blocks[-1]
            elif i == filled and self.level > 0:
                meter += blocks[int(self.level * 4) % len(blocks)]
            else:
                meter += blocks[0]

        return f"[dim]{meter}[/]"
