"""Launches main.py and dashboard.py together and manages them as one unit
- press Ctrl+C once to stop both cleanly.

Each one still runs standalone too (`uv run main.py` on its own works
fine, and dashboard.py reconnects automatically if Gabbie isn't up yet) -
this is just a convenience for running both at once.
"""

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


def stream_output(name, proc):
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)


def _handle_sigterm(signum, frame):
    # Python's default SIGTERM handling bypasses try/finally entirely, so
    # without this a plain `kill <pid>` would orphan the child processes
    # instead of running the cleanup below.
    raise SystemExit


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    procs = []
    try:
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

        # If any one of them exits on its own (e.g. crashes), bring the
        # whole suite down rather than leaving the others running solo.
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[suite] {name} exited ({code}), shutting down the rest")
                    raise SystemExit
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        print("[suite] shutting down...")
    finally:
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


if __name__ == "__main__":
    main()
