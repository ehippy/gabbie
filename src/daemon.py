"""Daemon process for Gabbie voice gateway."""

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from src.config import Config
from src.voice_gateway import VoiceGatewayService, GatewayEvent

logger = logging.getLogger(__name__)


class DaemonIPCServer:
    """IPC server for daemon-TUI communication."""

    def __init__(self, socket_path: str):
        self.socket_path = os.path.expanduser(socket_path)
        self._server_socket: socket.socket | None = None
        self._running = False
        self._clients: list[socket.socket] = []
        self._event_thread: threading.Thread | None = None
        self._gateway: VoiceGatewayService | None = None

    def start(self, gateway: VoiceGatewayService) -> None:
        """Start the IPC server."""
        self._gateway = gateway
        self._running = True

        # Remove existing socket file
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Create socket directory
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)

        # Create Unix socket
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self.socket_path)
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)

        logger.info(f"IPC server started on {self.socket_path}")

        # Start event forwarding thread
        self._event_thread = threading.Thread(target=self._forward_events, daemon=True)
        self._event_thread.start()

        # Accept clients
        while self._running:
            try:
                client, _ = self._server_socket.accept()
                client.settimeout(1.0)
                self._clients.append(client)
                logger.info(f"Client connected, total clients: {len(self._clients)}")
                # Start a thread to handle this client
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error accepting client: {e}")

    def stop(self) -> None:
        """Stop the IPC server."""
        self._running = False

        # Close all clients
        for client in self._clients:
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()

        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        # Remove socket file
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except Exception:
                pass

        logger.info("IPC server stopped")

    def _forward_events(self) -> None:
        """Forward gateway events to all connected clients."""
        while self._running:
            try:
                event = self._gateway.get_event(block=False)
                if event:
                    self._broadcast(
                        {
                            "type": "event",
                            "event_type": event.event_type,
                            "data": event.data,
                            "timestamp": event.timestamp,
                        }
                    )
                time.sleep(0.05)
            except Exception:
                continue

    def _broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        data = json.dumps(message).encode() + b"\n"
        dead_clients = []

        for client in self._clients:
            try:
                client.sendall(data)
            except Exception:
                dead_clients.append(client)

        # Remove dead clients
        for client in dead_clients:
            try:
                client.close()
                self._clients.remove(client)
            except Exception:
                pass

    def _handle_client(self, client: socket.socket) -> None:
        """Handle a client connection."""
        client.settimeout(5.0)
        buffer = b""

        while self._running:
            try:
                data = client.recv(4096)
                if not data:
                    break

                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    message = json.loads(line.decode())
                    response = self._handle_command(message)
                    client.sendall((json.dumps(response) + "\n").encode())

            except socket.timeout:
                continue
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error handling client: {e}")
                break

        try:
            client.close()
            if client in self._clients:
                self._clients.remove(client)
        except Exception:
            pass

    def _handle_command(self, message: dict) -> dict:
        """Handle a command from client."""
        command = message.get("command")

        if command == "start":
            self._gateway.start()
            return {"status": "ok", "message": "Gateway started"}

        elif command == "stop":
            self._gateway.stop()
            return {"status": "ok", "message": "Gateway stopped"}

        elif command == "get_status":
            return {"status": "ok", "data": self._gateway.get_status()}

        elif command == "list_devices":
            from src.audio_manager import AudioManager

            with AudioManager() as manager:
                devices = manager.list_devices()
            return {"status": "ok", "data": devices}

        elif command == "update_config":
            config = message.get("config", {})
            self._gateway.update_config(**config)
            return {"status": "ok", "message": "Config updated"}

        elif command == "get_config":
            return {"status": "ok", "data": self._gateway.config.to_dict()}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}


class Daemon:
    """Gabbie daemon process."""

    def __init__(self, config: Config):
        self.config = config
        self.gateway = VoiceGatewayService(config)
        self.ipc_server = DaemonIPCServer(config.daemon_socket_path)
        self.pid_file = os.path.expanduser(config.pid_file)
        self._running = False

    def start(self) -> None:
        """Start the daemon."""
        if self._is_running():
            logger.error("Daemon is already running")
            sys.exit(1)

        self._write_pid()

        self._running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start IPC server (blocks)
        try:
            self.ipc_server.start(self.gateway)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the daemon."""
        logger.info("Shutting down daemon...")
        self._running = False

        self.ipc_server.stop()
        self.gateway.stop()
        self._remove_pid()

        logger.info("Daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _is_running(self) -> bool:
        """Check if daemon is already running."""
        if not os.path.exists(self.pid_file):
            return False

        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, FileNotFoundError):
            return False

    def _write_pid(self) -> None:
        """Write PID file."""
        Path(self.pid_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if os.path.exists(self.pid_file):
            try:
                os.remove(self.pid_file)
            except Exception:
                pass


def main():
    """Main entry point for daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Gabbie Voice Gateway Daemon")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--log", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config.load(args.config)
    daemon = Daemon(config)
    daemon.start()


if __name__ == "__main__":
    main()
