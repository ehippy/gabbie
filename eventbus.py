"""Broadcasts Gabbie's state as JSON-lines events to any connected process
(the dashboard, or whatever else wants to watch) over a local TCP socket.
Optional side channel - if nothing's listening, or the port's unavailable,
Gabbie still runs fine without it.
"""

import json
import socket
import threading
from datetime import datetime

HOST = "127.0.0.1"
PORT = 8765


class EventBus:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server = None
        self.clients = []
        self.lock = threading.Lock()
        self.enabled = False

    def start(self):
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(5)
        except OSError:
            self.enabled = False
            return False
        self.enabled = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return True

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with self.lock:
                self.clients.append(conn)

    def publish(self, event, **data):
        if not self.enabled:
            return
        payload = json.dumps({"event": event, "ts": datetime.now().isoformat(), **data}) + "\n"
        line = payload.encode()
        with self.lock:
            dead = []
            for conn in self.clients:
                try:
                    conn.sendall(line)
                except OSError:
                    dead.append(conn)
            for conn in dead:
                self.clients.remove(conn)
