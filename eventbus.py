"""Broadcasts Gabbie's state as JSON-lines events to any connected process
(the dashboard, or whatever else wants to watch) over a local TCP socket.
Optional side channel - if nothing's listening, or the port's unavailable,
Gabbie still runs fine without it.

Also mirrors a coarse summary (state/task) to crumb's Mosquitto broker on
`axon/agents/gabbie/status` - the MQTT contract axon-native's AGENTS screen
and the WebXR deck subscribe to. Same optional-side-channel ethos: paho
missing, broker down, or publish error and the mirror just goes quiet.
"""

import json
import socket
import threading
from datetime import datetime

HOST = "127.0.0.1"
PORT = 8765

MQTT_BROKER = "crumb.local"
MQTT_PORT = 1883
STATUS_TOPIC = "axon/agents/gabbie/status"

# gabbie's rich event names -> the coarse {running, idle, error} contract.
# Ambient events (listening/transcribing/first_audio/herd noise) deliberately
# absent: they fire constantly from idle monitoring and would pin us to
# "running" forever - same lesson axon-native learned for its voice splash.
STATE_MAP = {
    "thinking": "running",
    "speaking": "running",
    "confirming": "running",
    "done_speaking": "idle",
    "sleeping": "idle",
    "bye": "idle",
    "error": "error",
}


class StatusMirror:
    """Best-effort publisher of {state, task} to the agent-status topic."""

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def start(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            return
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                 client_id="gabbie-status")
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            self._client = client
        except OSError:
            self._client = None

    def publish(self, state, task):
        with self._lock:
            if self._client is None:
                self.start()
                if self._client is None:
                    return
            payload = json.dumps({"state": state, "task": task})
            # retained so a restarting dashboard shows gabbie immediately
            try:
                self._client.publish(STATUS_TOPIC, payload, retain=True)
            except Exception:
                pass  # status must never break conversation


class EventBus:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server = None
        self.clients = []
        self.lock = threading.Lock()
        self.enabled = False
        self.mirror = StatusMirror()

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
        state = STATE_MAP.get(event)
        if state is not None:
            if state == "idle":
                task = "humming" if event == "done_speaking" else "sleeping"
            elif event == "confirming":
                task = (data.get("description") or "needs confirmation")[:80]
            else:
                task = (data.get("text") or event)[:80]
            self.mirror.publish(state, task)
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
