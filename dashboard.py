"""Local dashboard: browse/delete remembered facts, browse past
transcripts, and watch Gabbie's live status - all served from your own
machine, nothing leaves it.

Run with `uv run dashboard.py`, then open http://localhost:8766. Works
whether or not `uv run main.py` is currently running - it just shows
"disconnected" for live status until it is.
"""

import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from eventbus import HOST as GABBIE_HOST, PORT as GABBIE_PORT
from main import MEMORY_PATH, TRANSCRIPTS_DIR, load_memory

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8766
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

# One connection to Gabbie's event bus, fanned out to every open browser
# tab's live-status stream, so N tabs don't mean N connections to Gabbie.
subscribers = []
subscribers_lock = threading.Lock()


def eventbus_bridge():
    while True:
        try:
            sock = socket.create_connection((GABBIE_HOST, GABBIE_PORT), timeout=5)
            buffer = ""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    with subscribers_lock:
                        for q in subscribers:
                            q.put(line)
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass
        time.sleep(1)  # Gabbie not running yet, or connection dropped - retry


def remove_fact(ts):
    facts = load_memory()
    remaining = [f for f in facts if f.get("ts") != ts]
    removed = len(remaining) != len(facts)
    if removed:
        MEMORY_PATH.write_text(json.dumps(remaining, indent=2))
    return removed


def list_transcripts():
    if not TRANSCRIPTS_DIR.exists():
        return []
    return sorted((p.name for p in TRANSCRIPTS_DIR.glob("*.jsonl")), reverse=True)


def read_transcript(name):
    # Only serve filenames that actually exist in the transcripts dir -
    # blocks path traversal via a crafted "../" name.
    if name not in list_transcripts():
        return None
    entries = []
    for line in (TRANSCRIPTS_DIR / name).read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet - don't spam the terminal with request logs

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/memory":
            self._send_json(load_memory())
            return

        if path == "/api/transcripts":
            self._send_json(list_transcripts())
            return

        if path.startswith("/api/transcripts/"):
            name = unquote(path[len("/api/transcripts/") :])
            entries = read_transcript(name)
            if entries is None:
                self._send_json({"error": "not found"}, status=404)
            else:
                self._send_json(entries)
            return

        if path == "/api/status/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = queue.Queue()
            with subscribers_lock:
                subscribers.append(q)
            try:
                while True:
                    line = q.get()
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)
            return

        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/memory/"):
            ts = unquote(path[len("/api/memory/") :])
            self._send_json({"removed": remove_fact(ts)})
            return
        self.send_response(404)
        self.end_headers()


def main():
    threading.Thread(target=eventbus_bridge, daemon=True).start()
    server = ThreadingHTTPServer((DASHBOARD_HOST, DASHBOARD_PORT), Handler)
    print(f"Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
