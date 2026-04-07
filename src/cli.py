"""Command-line interface for Gabbie."""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import typer

from src.config import Config
from src.audio_manager import AudioManager

app = typer.Typer(name="gabbie", help="Gabbie Voice Gateway CLI")
logger = logging.getLogger(__name__)


def get_pid_file() -> str:
    """Get path to PID file."""
    return os.path.expanduser("~/.gabbie/gabbie.pid")


def get_socket_path() -> str:
    """Get path to Unix socket."""
    return os.path.expanduser("~/.gabbie/gabbie.sock")


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    pid_file = get_pid_file()
    if not os.path.exists(pid_file):
        return False

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, FileNotFoundError):
        return False


def send_command(command: dict) -> dict | None:
    """Send command to daemon via Unix socket."""
    import json
    import socket

    socket_path = get_socket_path()
    if not os.path.exists(socket_path):
        typer.echo(f"Error: Daemon not running. Start with 'gabbie daemon start'")
        return None

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        client.settimeout(10)

        data = json.dumps(command).encode() + b"\n"
        client.sendall(data)

        response = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk
            if response.endswith(b"\n"):
                break

        client.close()
        return json.loads(response.decode())

    except Exception as e:
        typer.echo(f"Error communicating with daemon: {e}")
        return None


@app.command()
def version():
    """Show version information."""
    from src import __version__

    typer.echo(f"Gabbie {__version__}")


@app.command()
def devices():
    """List available audio devices."""
    logging.basicConfig(level=logging.WARNING)

    typer.echo("Enumerating audio devices...")
    typer.echo("=" * 60)

    with AudioManager() as manager:
        devices = manager.list_devices()

        typer.echo(f"\nDEFAULT INPUT (index {devices['default_input']}):")
        for d in devices["input"]:
            if d["index"] == devices["default_input"]:
                typer.echo(
                    f"  [bold green]{d['index']}[/] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz) <-- DEFAULT"
                )
                break

        typer.echo(f"\nDEFAULT OUTPUT (index {devices['default_output']}):")
        for d in devices["output"]:
            if d["index"] == devices["default_output"]:
                typer.echo(
                    f"  [bold green]{d['index']}[/] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz) <-- DEFAULT"
                )
                break

        typer.echo(f"\nINPUT DEVICES ({len(devices['input'])} found):")
        for d in devices["input"]:
            marker = " <-- DEFAULT" if d["index"] == devices["default_input"] else ""
            color = "green" if d["index"] == devices["default_input"] else "white"
            typer.echo(
                f"  [{color}]{d['index']}[/] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz){marker}"
            )

        typer.echo(f"\nOUTPUT DEVICES ({len(devices['output'])} found):")
        for d in devices["output"]:
            marker = " <-- DEFAULT" if d["index"] == devices["default_output"] else ""
            color = "green" if d["index"] == devices["default_output"] else "white"
            typer.echo(
                f"  [{color}]{d['index']}[/] {d['name']} ({d['channels']}ch, {d['sample_rate']}Hz){marker}"
            )

    typer.echo("=" * 60)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
    reset: bool = typer.Option(False, "--reset", help="Reset config to defaults"),
    output_device: int | None = typer.Option(
        None, "--output", help="Set output device index"
    ),
    input_device: int | None = typer.Option(
        None, "--input", help="Set input device index"
    ),
    server_url: str | None = typer.Option(None, "--server", help="Set server URL"),
):
    """Manage configuration."""
    config_path = os.path.expanduser("~/.gabbie/config.toml")

    if reset:
        config = Config()
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        config.save(config_path)
        typer.echo(f"Config reset to defaults and saved to {config_path}")
        return

    if show:
        if not os.path.exists(config_path):
            typer.echo("No config file found. Creating default...")
            config = Config()
            config.save(config_path)

        config = Config.load(config_path)
        typer.echo(f"Config file: {config_path}")
        typer.echo("=" * 60)
        typer.echo(f"Input device:  {config.input_device_index}")
        typer.echo(f"Output device: {config.output_device_index}")
        typer.echo(f"Server URL:    {config.server_url}")
        typer.echo(f"Wake word:     {config.wake_word}")
        typer.echo(f"STT model:     {config.stt_model}")
        typer.echo(f"LLM model:     {config.llm_model}")
        typer.echo(f"TTS model:     {config.tts_model}")
        typer.echo(f"TTS voice:     {config.tts_voice}")
        typer.echo("=" * 60)
        return

    # Update config
    if not os.path.exists(config_path):
        config = Config()
    else:
        config = Config.load(config_path)

    changed = False
    if output_device is not None:
        config.output_device_index = output_device
        typer.echo(f"Set output device to {output_device}")
        changed = True
    if input_device is not None:
        config.input_device_index = input_device
        typer.echo(f"Set input device to {input_device}")
        changed = True
    if server_url is not None:
        config.server_url = server_url
        typer.echo(f"Set server URL to {server_url}")
        changed = True

    if changed:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        config.save(config_path)
        typer.echo(f"Config saved to {config_path}")
    else:
        typer.echo(
            "No changes made. Use --show to view config or specify options to update."
        )


daemon_app = typer.Typer(name="daemon", help="Daemon management commands")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command()
def start(
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in background"),
    log_level: str = typer.Option("INFO", "--log", "-l", help="Log level"),
):
    """Start the daemon."""
    if is_daemon_running():
        typer.echo("Daemon is already running")
        return

    # Start daemon process
    cmd = [sys.executable, "-m", "src.daemon", "--log", log_level]

    if detach:
        # Run in background
        with open(os.devnull, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f, start_new_session=True)
        typer.echo(f"Daemon started (PID {process.pid})")
    else:
        # Run in foreground
        typer.echo("Starting daemon (Ctrl+C to stop)...")
        os.execvp(sys.executable, cmd)


@daemon_app.command()
def stop():
    """Stop the daemon."""
    pid_file = get_pid_file()

    if not os.path.exists(pid_file):
        typer.echo("Daemon is not running")
        return

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        os.kill(pid, signal.SIGTERM)
        typer.echo(f"Daemon stopped (PID {pid})")

        # Wait for PID file to be removed
        for _ in range(10):
            if not os.path.exists(pid_file):
                break
            time.sleep(0.1)

    except ProcessLookupError:
        typer.echo("Daemon process not found, cleaning up PID file")
        os.remove(pid_file)
    except Exception as e:
        typer.echo(f"Error stopping daemon: {e}")


@daemon_app.command()
def status():
    """Check daemon status."""
    if is_daemon_running():
        typer.echo("Daemon is running")

        # Get status from daemon
        response = send_command({"command": "get_status"})
        if response and response.get("status") == "ok":
            data = response.get("data", {})
            typer.echo(f"  State: {data.get('state', 'unknown')}")
            typer.echo(f"  Conversations: {data.get('conversation_count', 0)}")
    else:
        typer.echo("Daemon is not running")
        typer.echo("Start with: gabbie daemon start")


@daemon_app.command()
def restart():
    """Restart the daemon."""
    if is_daemon_running():
        stop()
        time.sleep(0.5)
    start()


@app.command()
def tui():
    """Launch the TUI interface."""
    if not is_daemon_running():
        typer.echo("Daemon not running. Starting daemon...")
        cmd = [sys.executable, "-m", "src.daemon"]
        process = subprocess.Popen(cmd, start_new_session=True)
        time.sleep(1)

    from src.tui.app import main as tui_main

    tui_main()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
):
    """Gabbie Voice Gateway - Start TUI (auto-starts daemon if needed)."""
    if ctx.invoked_subcommand is None:
        tui()


def run():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    run()
