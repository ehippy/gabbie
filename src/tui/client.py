"""IPC client for TUI-daemon communication."""

import json
import logging
import os
import socket
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DaemonClient:
    """Client for communicating with Gabbie daemon."""

    def __init__(self, socket_path: str | None = None):
        self.socket_path = os.path.expanduser(socket_path or "~/.gabbie/gabbie.sock")
        self._client: socket.socket | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Check if connected to daemon."""
        return self._connected and os.path.exists(self.socket_path)

    def connect(self) -> bool:
        """Connect to daemon."""
        if not os.path.exists(self.socket_path):
            return False

        try:
            self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._client.connect(self.socket_path)
            self._client.settimeout(10)
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to daemon: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from daemon."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False

    def send_command(self, command: str, **kwargs) -> dict[str, Any] | None:
        """Send command to daemon."""
        if not self._connected:
            if not self.connect():
                return None

        message = {"command": command, **kwargs}

        try:
            data = json.dumps(message).encode() + b"\n"
            self._client.sendall(data)
            logger.debug(f"Sent command: {command}")

            response = b""
            while True:
                chunk = self._client.recv(4096)
                if not chunk:
                    if response:
                        logger.warning(
                            f"Connection closed but got response: {response}"
                        )
                    break
                response += chunk
                if response.endswith(b"\n"):
                    break

            if not response:
                logger.error(f"Empty response for command: {command}")
                return None

            response_str = response.decode().strip()
            logger.debug(f"Received response: {response_str}")

            lines = response_str.split("\n")
            for line in lines:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and data.get("status") in ("ok", "error"):
                        return data
                except json.JSONDecodeError:
                    continue

            logger.error(f"No valid response found in: {response_str}")
            return None

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decode error for command {command}: {e}. Raw response: {response}"
            )
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Command failed: {e}")
            self._connected = False
            return None

    def get_status(self) -> dict[str, Any] | None:
        """Get gateway status."""
        return self.send_command("get_status")

    def start(self) -> dict[str, Any] | None:
        """Start voice gateway."""
        return self.send_command("start")

    def stop(self) -> dict[str, Any] | None:
        """Stop voice gateway."""
        return self.send_command("stop")

    def list_devices(self) -> dict[str, Any] | None:
        """List audio devices."""
        return self.send_command("list_devices")

    def update_config(self, **config) -> dict[str, Any] | None:
        """Update configuration."""
        return self.send_command("update_config", config=config)

    def get_config(self) -> dict[str, Any] | None:
        """Get current configuration."""
        return self.send_command("get_config")

    def get_events(self) -> dict[str, Any] | None:
        """Drain pending gateway events."""
        return self.send_command("get_events")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
