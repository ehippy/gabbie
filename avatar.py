"""Bare-bones placeholder avatar - proves out the event plumbing with a
colored circle and a state label. Swap the drawing code for something real
once the mechanism's confirmed working; the event-handling loop below stays
the same regardless of what you draw.

Run this in a separate terminal alongside `uv run main.py`. It reconnects
automatically if Gabbie isn't running yet or the connection drops.
"""

import json
import queue
import socket
import threading
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

import pygame

from eventbus import HOST, PORT

WINDOW_SIZE = (320, 360)
STATE_COLORS = {
    "idle": (55, 55, 65),
    "listening": (70, 190, 100),
    "transcribing": (70, 190, 100),
    "thinking": (230, 180, 40),
    "speaking": (70, 130, 230),
    "sleeping": (90, 60, 140),
}


def connect_loop(event_queue, connected_flag):
    while True:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=5)
            connected_flag["connected"] = True
            buffer = ""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        event_queue.put(json.loads(line))
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass
        connected_flag["connected"] = False
        time.sleep(1)  # Gabbie not running yet, or connection dropped - retry


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Gabbie")
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    event_queue = queue.Queue()
    connected_flag = {"connected": False}
    threading.Thread(target=connect_loop, args=(event_queue, connected_flag), daemon=True).start()

    state = "idle"
    last_heard = ""
    last_said = ""

    running = True
    try:
        while running:
            for pg_event in pygame.event.get():
                if pg_event.type == pygame.QUIT:
                    running = False

            while not event_queue.empty():
                evt = event_queue.get()
                kind = evt.get("event")
                if kind in ("listening", "transcribing", "thinking", "sleeping"):
                    state = kind
                elif kind == "heard":
                    if not evt.get("ignored"):
                        last_heard = evt["text"]
                elif kind == "speaking":
                    state = "speaking"
                    last_said = evt["text"]
                elif kind == "done_speaking":
                    state = "listening"
                elif kind == "bye":
                    state = "idle"

            screen.fill((18, 18, 22))
            color = STATE_COLORS.get(state, (100, 100, 100))
            pygame.draw.circle(screen, color, (160, 140), 80)

            label = font.render(state, True, (235, 235, 235))
            screen.blit(label, (160 - label.get_width() // 2, 235))

            conn_text = "connected" if connected_flag["connected"] else "waiting for gabbie..."
            conn_color = (120, 200, 120) if connected_flag["connected"] else (200, 120, 120)
            screen.blit(small_font.render(conn_text, True, conn_color), (10, 10))

            if last_heard:
                screen.blit(small_font.render(f"You: {last_heard}"[:44], True, (170, 170, 170)), (10, 280))
            if last_said:
                screen.blit(small_font.render(f"Gabbie: {last_said}"[:44], True, (170, 170, 170)), (10, 305))

            pygame.display.flip()
            clock.tick(30)
    except KeyboardInterrupt:
        pass

    pygame.quit()


if __name__ == "__main__":
    main()
