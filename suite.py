"""Launches main.py and dashboard.py together and manages them as one unit
- press Ctrl+C once to stop both cleanly.

Each one still runs standalone too (`uv run main.py` on its own works
fine, and dashboard.py reconnects automatically if Gabbie isn't up yet) -
this is just a convenience for running both at once.

Pass --watch to also restart both whenever a .py file in the project
changes, so you don't have to stop/start the suite by hand while
iterating: `uv run suite.py --watch`.
"""

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSES = [
    ("main", "main.py"),
    ("dashboard", "dashboard.py"),
]
SHUTDOWN_TIMEOUT_SECONDS = 5
WATCH_POLL_SECONDS = 1.0
WATCH_QUIET_SECONDS = 0.3  # let a multi-file save settle before restarting


def stream_output(name, proc):
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)


def _handle_sigterm(signum, frame):
    # Python's default SIGTERM handling bypasses try/finally entirely, so
    # without this a plain `kill <pid>` would orphan the child processes
    # instead of running the cleanup below.
    raise SystemExit


def launch_processes():
    procs = []
    for name, script in PROCESSES:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BASE_DIR / script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BASE_DIR,
        )
        procs.append((name, proc))
        threading.Thread(target=stream_output, args=(name, proc), daemon=True).start()
    return procs


def stop_processes(procs):
    for name, proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + SHUTDOWN_TIMEOUT_SECONDS
    for name, proc in procs:
        try:
            proc.wait(timeout=max(0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            print(f"[suite] {name} didn't exit in time, killing it")
            proc.kill()


def _snapshot_mtimes():
    # Flat project, no subdirs worth watching - .venv/transcripts/etc are
    # excluded just by not globbing into them.
    return {p: p.stat().st_mtime for p in BASE_DIR.glob("*.py")}


def watch_for_changes(restart_event, stop_event):
    known = _snapshot_mtimes()
    while not stop_event.is_set():
        time.sleep(WATCH_POLL_SECONDS)
        current = _snapshot_mtimes()
        if current == known:
            continue
        # Debounce: a save can touch several files (or a formatter can
        # rewrite one twice) - wait for things to go quiet before firing.
        while True:
            time.sleep(WATCH_QUIET_SECONDS)
            settled = _snapshot_mtimes()
            if settled == current:
                break
            current = settled
        known = current
        restart_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch",
        action="store_true",
        help="restart main.py/dashboard.py whenever a .py file in the project changes",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    stop_event = threading.Event()
    restart_event = threading.Event()
    if args.watch:
        threading.Thread(target=watch_for_changes, args=(restart_event, stop_event), daemon=True).start()
        print("[suite] watching for .py changes - save a file to restart")

    procs = []
    try:
        while True:
            procs = launch_processes()
            restart_event.clear()

            # If any one of them exits on its own (e.g. crashes), bring the
            # whole suite down rather than leaving the others running solo.
            while True:
                if restart_event.is_set():
                    print("[suite] code changed, restarting...")
                    stop_processes(procs)
                    break
                exited = next((p for p in procs if p[1].poll() is not None), None)
                if exited:
                    name, proc = exited
                    print(f"[suite] {name} exited ({proc.returncode}), shutting down the rest")
                    raise SystemExit
                time.sleep(0.5)

            if not args.watch:
                break
    except (KeyboardInterrupt, SystemExit):
        print("[suite] shutting down...")
    finally:
        stop_event.set()
        stop_processes(procs)


if __name__ == "__main__":
    main()
