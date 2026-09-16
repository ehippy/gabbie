import collections
import html.parser
import io
import json
import os
import queue
import random
import re
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

import httpx
from faster_whisper import WhisperModel
from openai import OpenAI
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

from eventbus import EventBus

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
PADDING_FRAMES = 10  # ~300ms of context on either side of speech
START_RATIO = 0.6  # fraction of padding frames that must be voiced to start
END_RATIO = 0.7  # fraction of padding frames that must be unvoiced to stop
VAD_MODE = 2  # 0 (permissive) - 3 (aggressive)
MIN_UTTERANCE_SECONDS = 0.3
PLAYBACK_COOLDOWN_SECONDS = 0.3
AWAKE_WINDOW_SECONDS = 20  # how long a conversation stays wake-word-free

NEURALFORGE_URL = "http://neuralforge:13305/v1"
LLM_MODEL = "Qwen3.8-27B-GGUF-UD-Q4_K_XL"
TTS_MODEL = "kokoro-v1"
TTS_VOICE = "af_heart"
SYSTEM_PROMPT = (
    "You are Gabbie, a warm, playful voice assistant, speaking out loud in "
    "a live conversation. Match your reply length to the moment: a simple "
    "yes/no or acknowledgment deserves just a word or short phrase ('Yep!', "
    "'Nope, not really.', 'Got it!'), not a full sentence. Never use more "
    "than two short sentences total, and only reach two when there's "
    "genuinely more to say. Don't tack on a follow-up question out of habit "
    "- only ask one when it's natural. Only go longer than two sentences "
    "when the user explicitly asks for detail, a list, or an explanation. "
    "Being brief does NOT mean sounding cold, flat, or robotic - stay warm "
    "and playful even in a one-word reply. When the user shares something "
    "personal or new about themselves, warmly acknowledge THAT SPECIFIC "
    "thing in your own words, briefly - never a flat, clinical 'Noted' or "
    "'Fair', and never a stock phrase or an unrelated fact from earlier in "
    "the conversation. You do have long-term memory: important facts get saved "
    "automatically and recalled in future conversations. If you don't know "
    "something about the user yet, say so plainly rather than claiming you "
    "have no memory at all. You also have tools to check the current date/"
    "time, list/read files on the user's computer, search the web, fetch a "
    "specific web page, search for images, and write or edit files. Image "
    "search shows results on the local dashboard, not out loud - you can't "
    "see the images either, so after calling it just acknowledge that "
    "you've pulled them up, don't invent a description of what's in them. "
    "When you use a tool, "
    "summarize what's useful from the result in your own words rather than "
    "reciting it verbatim - especially file contents or web pages, which "
    "can be long. Writing or editing a file always asks the user to "
    "confirm out loud before it actually happens, automatically - so just "
    "call the tool directly when asked to write or edit something, don't "
    "ask for confirmation yourself first, that would just double up."
)
MEMORY_EXTRACTION_PROMPT = (
    "You maintain Gabbie's long-term memory about the user. You'll be given "
    "the CURRENT remembered facts and a recent conversation excerpt. Decide "
    "what should change:\n"
    '- "add": new durable facts worth remembering long-term (name, '
    "preferences, ongoing projects, decisions) that aren't already covered "
    "by a current fact. Use the excerpt for context to make facts "
    "self-contained (e.g. resolve 'they' or 'nine and eleven' to what it "
    "refers to).\n"
    '- "remove": any CURRENT facts this conversation shows are now wrong, '
    "corrected, or outdated. Copy them EXACTLY as given so they can be "
    "matched.\n"
    'Reply with ONLY JSON: {"add": [...], "remove": [...]}. Use empty '
    "arrays for either if there's nothing to add or remove."
)
BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
MEMORY_PATH = BASE_DIR / "memory.json"
GOODBYE_WORDS = ("goodbye", "bye", "exit", "quit")
GOODBYE_PATTERN = re.compile(r"\b(" + "|".join(GOODBYE_WORDS) + r")\b", re.IGNORECASE)
SLEEP_PHRASES = ("stop listening", "go to sleep", "go back to sleep")
# Checked in this order (negative first) when confirming a destructive
# action - "no" wins over any accidental affirmative-sounding word nearby.
NEGATIVE_WORDS = ("no", "nope", "don't", "do not", "stop", "cancel", "negative", "never mind")
AFFIRMATIVE_WORDS = ("yes", "yeah", "yep", "yup", "sure", "go ahead", "do it", "confirm", "affirmative", "okay", "ok")
NEGATIVE_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in NEGATIVE_WORDS) + r")\b", re.IGNORECASE)
AFFIRMATIVE_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in AFFIRMATIVE_WORDS) + r")\b", re.IGNORECASE)
# faster-whisper hears "Gabbie" a few different ways in practice.
WAKE_WORDS = ("gabbie", "gabby", "gabi", "gaby")
WAKE_PATTERN = re.compile(r"\b(" + "|".join(WAKE_WORDS) + r")\b", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"[.!?]+\s+")


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}")


def make_input_callback(audio_queue, listening_enabled):
    def callback(indata, frame_count, time_info, status):
        if listening_enabled.is_set():
            audio_queue.put(indata.tobytes())

    return callback


def drain_queue(audio_queue):
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def listen_for_utterance(vad, audio_queue):
    ring_buffer = collections.deque(maxlen=PADDING_FRAMES)
    triggered = False
    voiced_frames = []

    while True:
        frame = audio_queue.get()
        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            voiced = sum(1 for f, s in ring_buffer if s)
            if voiced > START_RATIO * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(f for f, s in ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            unvoiced = sum(1 for f, s in ring_buffer if not s)
            if unvoiced > END_RATIO * ring_buffer.maxlen:
                break

    pcm = b"".join(voiced_frames)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def contains_wake_word(text):
    return bool(WAKE_PATTERN.search(text))


def contains_sleep_phrase(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in SLEEP_PHRASES)


def contains_negative(text):
    return bool(NEGATIVE_PATTERN.search(text))


def contains_affirmative(text):
    return bool(AFFIRMATIVE_PATTERN.search(text))


def transcribe(whisper, audio):
    if audio.size == 0:
        return ""
    segments, _ = whisper.transcribe(
        audio, language="en", initial_prompt="Gabbie"
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize(client, text):
    response = client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
    )
    return sf.read(io.BytesIO(response.content), dtype="float32")


def load_memory():
    if not MEMORY_PATH.exists():
        return []
    try:
        return json.loads(MEMORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def build_system_prompt(facts):
    if not facts:
        return SYSTEM_PROMPT
    bullets = "\n".join(f"- {f['fact']}" for f in facts)
    return f"{SYSTEM_PROMPT}\n\nThings you remember from previous conversations:\n{bullets}"


def open_transcript():
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}.jsonl"
    return path


def append_transcript(path, role, content):
    entry = {"ts": datetime.now().isoformat(), "role": role, "content": content}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def extract_memory_updates(client, current_facts, recent_messages):
    # A single isolated exchange often doesn't make sense on its own (e.g.
    # "Nine and eleven" only means something next to "how old are they?"),
    # so extraction gets a few turns of context instead of just the latest.
    # It also gets the current facts so it can catch contradictions/updates
    # (e.g. a correction) instead of just piling on duplicates forever.
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gabbie'}: {m['content']}"
        for m in recent_messages
        # A tool-calling turn's assistant message often has content=None
        # (the model went straight for the tool with nothing to say first)
        # - skip those rather than feeding the extractor literal "Gabbie:
        # None" lines.
        if m["role"] in ("user", "assistant") and m.get("content")
    )
    current_block = "\n".join(f"- {f}" for f in current_facts) or "(none yet)"
    user_content = f"Current known facts:\n{current_block}\n\nRecent conversation:\n{convo}"
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            # Unlike the conversational reply path, this runs off the
            # latency-critical path (see the join() in main()), so it's
            # worth leaving "thinking" enabled here - resolving something
            # like "nine and eleven" back to "the kids' ages" needs it.
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        add = [f.strip() for f in data.get("add", []) if isinstance(f, str) and f.strip()]
        remove = [f.strip() for f in data.get("remove", []) if isinstance(f, str) and f.strip()]
        return add, remove
    except Exception:
        return [], []


def remember_worker(client, memory_lock, events, recent_messages):
    with memory_lock:
        current_facts = [f["fact"] for f in load_memory()]
    add, remove = extract_memory_updates(client, current_facts, recent_messages)
    if not add and not remove:
        return
    with memory_lock:
        existing = load_memory()
        if remove:
            remove_set = set(remove)
            for f in existing:
                if f["fact"] in remove_set:
                    log(f"[forgot] {f['fact']}")
                    events.publish("forgot", fact=f["fact"])
            existing = [f for f in existing if f["fact"] not in remove_set]
        known = {f["fact"] for f in existing}
        for fact in add:
            if fact not in known:
                existing.append({"fact": fact, "ts": datetime.now().isoformat()})
                known.add(fact)
                log(f"[remembered] {fact}")
                events.publish("remembered", fact=fact)
        MEMORY_PATH.write_text(json.dumps(existing, indent=2))


MAX_TOOL_ROUNDS = 3
MAX_FILE_CHARS = 4000
MAX_DIRECTORY_ENTRIES = 100
MAX_FETCH_CHARS = 6000
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"
# The model doesn't reliably say anything before calling a tool (sometimes
# it's dead silence straight into the tool call), and using a tool costs a
# full extra LLM round-trip - so guarantee an acknowledgment instead of
# leaving the user wondering if she heard them.
TOOL_FILLER_PHRASES = (
    "One sec, let me check.",
    "Hold on, checking that now.",
    "Let me take a look.",
    "Give me just a moment.",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory on "
            "the user's computer. Defaults to their home directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file on the user's computer.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Returns "
            "a short list of results (title, snippet, URL).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Search the web for images matching a query and "
            "display them on the local dashboard. There's no way to show a "
            "picture out loud, so only use this when the user asks to see, "
            "find, or look up pictures/photos/images of something - the "
            "result is visual only, not something to describe back.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Image search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a specific web page and return its readable text content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The http(s) URL to fetch"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a file or overwrite an existing one with new "
            "content on the user's computer. Always asks the user to confirm "
            "out loud first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Full text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact, unique piece of text in an "
            "existing file on the user's computer with new text. Always asks "
            "the user to confirm out loud first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_text": {"type": "string", "description": "Exact text to replace, must be unique in the file"},
                    "new_text": {"type": "string", "description": "Text to replace it with"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

# Tools that change something on disk - always confirmed out loud before
# they actually run, since voice input is lossy and there's no click-to-
# confirm UI to catch a mis-transcribed request before it does something
# irreversible.
DESTRUCTIVE_TOOLS = {"write_file", "edit_file"}


def _resolve_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path.home() / p


def tool_get_current_datetime(**_kwargs):
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


def tool_list_directory(path=None, **_kwargs):
    p = _resolve_path(path) if path else Path.home()
    try:
        entries = sorted(p.iterdir())
    except FileNotFoundError:
        return f"Error: no such directory: {p}"
    except NotADirectoryError:
        return f"Error: not a directory: {p}"
    except PermissionError:
        return f"Error: permission denied: {p}"
    if not entries:
        return f"{p} is empty."
    names = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_DIRECTORY_ENTRIES]]
    listing = f"Contents of {p}:\n" + "\n".join(names)
    if len(entries) > MAX_DIRECTORY_ENTRIES:
        listing += f"\n[...{len(entries) - MAX_DIRECTORY_ENTRIES} more not shown]"
    return listing


def tool_read_file(path=None, **_kwargs):
    if not path:
        return "Error: no path given"
    p = _resolve_path(path)
    try:
        text = p.read_text(errors="replace")
    except FileNotFoundError:
        return f"Error: no such file: {p}"
    except IsADirectoryError:
        return f"Error: {p} is a directory, not a file"
    except PermissionError:
        return f"Error: permission denied: {p}"
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + f"\n\n[...truncated, {len(text)} characters total]"
    return text


class _TextExtractor(html.parser.HTMLParser):
    """Bare-bones HTML-to-text: drops tags/script/style, keeps the rest.
    Good enough for an LLM to summarize from - not a real readability
    extractor, and doesn't need to be for that."""

    def __init__(self):
        super().__init__()
        self._skip = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)


def _html_to_text(html_content):
    parser = _TextExtractor()
    parser.feed(html_content)
    return "\n".join(parser.parts)


def tool_fetch_url(url=None, **_kwargs):
    if not url:
        return "Error: no url given"
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        return f"Error: unsupported URL scheme: {scheme or '(none)'}"
    try:
        response = httpx.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True
        )
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"
    if response.status_code >= 400:
        return f"Error: {url} returned HTTP {response.status_code}"
    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        text = _html_to_text(response.text).strip()
    elif "text" in content_type or not content_type:
        text = response.text.strip()
    else:
        return f"Error: {url} isn't a text/HTML page (content-type: {content_type})"
    if not text:
        return f"{url} returned no readable text content."
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + f"\n\n[...truncated, {len(text)} characters total]"
    return text, {"url": url}


def tool_web_search(query=None, **_kwargs):
    if not query:
        return "Error: no query given"
    if not BRAVE_API_KEY:
        return "Error: web search isn't set up yet - no BRAVE_API_KEY configured."
    try:
        response = httpx.get(
            BRAVE_SEARCH_URL,
            params={"q": query, "count": 5},
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            timeout=10,
        )
    except httpx.HTTPError as e:
        return f"Error searching: {e}"
    if response.status_code != 200:
        return f"Error: search returned HTTP {response.status_code}"
    results = response.json().get("web", {}).get("results", [])
    if not results:
        return f"No search results for '{query}'."
    lines = []
    structured = []
    for r in results[:5]:
        title = r.get("title", "")
        description = re.sub(r"<[^>]+>", "", r.get("description", ""))
        url = r.get("url", "")
        lines.append(f"- {title}: {description} ({url})")
        structured.append({"title": title, "description": description, "url": url})
    return "\n".join(lines), {"query": query, "results": structured}


def tool_image_search(query=None, **_kwargs):
    if not query:
        return "Error: no query given"
    if not BRAVE_API_KEY:
        return "Error: image search isn't set up yet - no BRAVE_API_KEY configured."
    try:
        response = httpx.get(
            BRAVE_IMAGE_SEARCH_URL,
            params={"q": query, "count": 20},
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            timeout=10,
        )
    except httpx.HTTPError as e:
        return f"Error searching images: {e}"
    if response.status_code != 200:
        return f"Error: image search returned HTTP {response.status_code}"
    results = response.json().get("results", [])
    structured = []
    for r in results[:20]:
        thumbnail = r.get("thumbnail", {}).get("src")
        if not thumbnail:
            continue  # nothing to show for this one, skip it
        structured.append(
            {
                "title": r.get("title", ""),
                "source_url": r.get("url", ""),
                "thumbnail": thumbnail,
                # Full-resolution image for the dashboard's lightbox - hosted
                # on the original site rather than Brave's thumbnail proxy,
                # so it's more likely to be hotlink-protected; the lightbox
                # falls back to the thumbnail if it fails to load.
                "image_url": r.get("properties", {}).get("url", ""),
            }
        )
    if not structured:
        return f"No image results for '{query}'."
    # The model can't actually see these - it's told just enough to
    # acknowledge the request without inventing a description of images it
    # never received.
    text = f"Found {len(structured)} images for '{query}' - shown on the dashboard, not spoken."
    return text, {"query": query, "results": structured}


def tool_write_file(path=None, content=None, **_kwargs):
    if not path:
        return "Error: no path given"
    if content is None:
        return "Error: no content given"
    p = _resolve_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except PermissionError:
        return f"Error: permission denied: {p}"
    except OSError as e:
        return f"Error writing {p}: {e}"
    return f"Wrote {len(content)} characters to {p}."


def tool_edit_file(path=None, old_text=None, new_text=None, **_kwargs):
    if not path:
        return "Error: no path given"
    if old_text is None or new_text is None:
        return "Error: need both old_text and new_text"
    p = _resolve_path(path)
    try:
        current = p.read_text()
    except FileNotFoundError:
        return f"Error: no such file: {p}"
    except PermissionError:
        return f"Error: permission denied: {p}"
    count = current.count(old_text)
    if count == 0:
        return f"Error: that exact text was not found in {p}"
    if count > 1:
        return f"Error: that text appears {count} times in {p} - it must be unique. Include more surrounding context."
    try:
        p.write_text(current.replace(old_text, new_text, 1))
    except PermissionError:
        return f"Error: permission denied: {p}"
    return f"Replaced the text in {p}."


def describe_tool_action(name, kwargs):
    path = kwargs.get("path", "(unknown path)")
    if name == "write_file":
        return f"I want to write to the file {path}."
    if name == "edit_file":
        return f"I want to edit the file {path}."
    return f"I want to run {name}."


TOOL_FUNCTIONS = {
    "get_current_datetime": tool_get_current_datetime,
    "list_directory": tool_list_directory,
    "read_file": tool_read_file,
    "web_search": tool_web_search,
    "image_search": tool_image_search,
    "fetch_url": tool_fetch_url,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}


def execute_tool(name, arguments_json):
    """Returns (text_for_the_model, extra_for_display). extra_for_display is
    None unless the tool has something worth showing visually (e.g. search
    results) - most tools just return a plain string, normalized here."""
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        kwargs = {}
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'", None
    try:
        result = func(**kwargs)
    except Exception as e:
        return f"Error running {name}: {e}", None
    if isinstance(result, tuple):
        return result
    return str(result), None


def speak(client, listening_enabled, audio_queue, events, text):
    """Speak a short, fixed line (no streaming needed)."""
    listening_enabled.clear()
    drain_queue(audio_queue)
    try:
        events.publish("speaking", text=text)
        audio, sr = synthesize(client, text)
        sd.play(audio, sr)
        sd.wait()
        time.sleep(PLAYBACK_COOLDOWN_SECONDS)
    finally:
        drain_queue(audio_queue)
        listening_enabled.set()


def confirm_with_user(client, listening_enabled, audio_queue, events, vad, whisper, description):
    """Speaks the proposed action and blocks until a clear yes/no comes
    back. Runs on the same thread that owns audio playback (see the
    "confirm" branch in think_and_speak's draining loop below) - never call
    this from a background thread, since sd.play() isn't safe to call
    concurrently from two threads at once."""
    prompt = f"{description} Should I go ahead?"
    log(f"Gabbie: {prompt}")
    events.publish("confirming", description=description)
    speak(client, listening_enabled, audio_queue, events, prompt)
    for _ in range(2):
        audio = listen_for_utterance(vad, audio_queue)
        if audio.size < SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
            continue
        text = transcribe(whisper, audio)
        if not text:
            continue
        log(f"[confirmation response] {text}")
        if contains_negative(text):
            return False
        if contains_affirmative(text):
            return True
        speak(client, listening_enabled, audio_queue, events, "Sorry, was that a yes or a no?")
    log("[confirmation] no clear answer, defaulting to no")
    return False


def think_and_speak(client, listening_enabled, audio_queue, events, vad, whisper, messages):
    """Stream the LLM reply, synthesizing and playing each sentence as soon
    as it's complete instead of waiting for the whole reply. Returns the
    full reply text once everything has finished playing."""
    listening_enabled.clear()
    drain_queue(audio_queue)
    try:
        sentence_queue = queue.Queue()
        clip_queue = queue.Queue(maxsize=2)
        full_text = {}

        def llm_worker():
            # This model, at the server's default temperature (1.0),
            # measurably stops prematurely (empty or 1-2 token replies) on a
            # meaningful fraction of turns - confirmed via a side-by-side
            # test: 5/10 suspect replies at temp=1.0 vs 0/10 at temp=0.6,
            # personality intact. Overriding it here for reliability.
            empty_retries_left = 2
            filler_spoken = False
            for _round in range(MAX_TOOL_ROUNDS + 2):
                parts = []
                buffer = ""
                tool_calls = {}
                llm_stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    stream=True,
                    temperature=0.6,
                    # This model "thinks" before answering by default, which
                    # roughly doubles latency for zero benefit on casual
                    # chat. It isn't always honored though - the raw
                    # <think>...</think> block sometimes leaks straight into
                    # content, so it's also stripped below as a safety net.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                for chunk in llm_stream:
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            entry = tool_calls.setdefault(
                                tc.index, {"id": None, "name": "", "arguments": ""}
                            )
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function and tc.function.name:
                                entry["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                entry["arguments"] += tc.function.arguments
                    content = delta.content
                    if not content:
                        continue
                    parts.append(content)
                    buffer += content
                    if "<think>" in buffer and "</think>" not in buffer:
                        continue  # mid leaked-reasoning block; wait for it to close
                    buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL)
                    match = SENTENCE_BOUNDARY.search(buffer)
                    while match:
                        sentence = buffer[: match.end()].strip()
                        buffer = buffer[match.end() :]
                        if sentence:
                            sentence_queue.put(sentence)
                        match = SENTENCE_BOUNDARY.search(buffer)
                # Guard against the stream ending mid-<think> (never closed):
                # don't speak or store the raw reasoning fragment.
                if "<think>" not in buffer and buffer.strip():
                    sentence_queue.put(buffer.strip())
                # Strip here too so a leaked <think> block never ends up in
                # conversation history or the transcript, even though it's
                # already kept out of what actually gets spoken above.
                full_reply = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.DOTALL)
                full_reply = re.sub(r"<think>.*$", "", full_reply, flags=re.DOTALL).strip()

                if tool_calls:
                    if not full_reply and not filler_spoken:
                        sentence_queue.put(random.choice(TOOL_FILLER_PHRASES))
                        filler_spoken = True
                    ordered = [tool_calls[i] for i in sorted(tool_calls)]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": full_reply or None,
                            "tool_calls": [
                                {
                                    "id": c["id"],
                                    "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]},
                                }
                                for c in ordered
                            ],
                        }
                    )
                    for c in ordered:
                        log(f"[tool] {c['name']}({c['arguments']})")
                        if c["name"] in DESTRUCTIVE_TOOLS:
                            try:
                                call_kwargs = json.loads(c["arguments"]) if c["arguments"] else {}
                            except json.JSONDecodeError:
                                call_kwargs = {}
                            description = describe_tool_action(c["name"], call_kwargs)
                            # Confirmation needs to speak and listen, which
                            # must happen on the thread that owns audio
                            # playback - not here (this is a background
                            # thread) - so hand it to the draining loop
                            # below and block for its answer. Routed through
                            # sentence_queue (not put on clip_queue directly)
                            # so tts_worker stays the sole producer into
                            # clip_queue - otherwise this, being near-instant,
                            # can race ahead of a still-synthesizing filler
                            # phrase queued moments earlier and get spoken
                            # (and confirmed/acted on) before it.
                            response_box = queue.Queue(maxsize=1)
                            sentence_queue.put(("confirm", description, response_box))
                            approved = response_box.get()
                            if approved:
                                result_text, result_extra = execute_tool(c["name"], c["arguments"])
                            else:
                                result_text, result_extra = (
                                    "The user did not confirm this action, so it was not performed.",
                                    None,
                                )
                        else:
                            result_text, result_extra = execute_tool(c["name"], c["arguments"])
                        events.publish(
                            "tool_call",
                            name=c["name"],
                            arguments=c["arguments"],
                            result=result_text,
                            extra=result_extra,
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": c["id"], "content": result_text}
                        )
                    continue  # go around again for the real answer now that tools ran

                full_text["reply"] = full_reply
                if full_reply or empty_retries_left <= 0:
                    break
                empty_retries_left -= 1
                log("[empty reply, retrying]")
            else:
                # Every round kept calling tools and the loop ran out
                # without ever reaching a final text answer - otherwise
                # this leaves the user with dead air after the filler
                # phrase (already spoken) and an empty turn in history.
                log("[tool round limit hit, giving up]")
                fallback = "Sorry, I'm having trouble finishing that - can you try again?"
                full_text["reply"] = fallback
                sentence_queue.put(fallback)
            sentence_queue.put(None)

        def tts_worker():
            while True:
                item = sentence_queue.get()
                if item is None:
                    clip_queue.put(None)
                    return
                if isinstance(item, tuple):
                    # Confirmation request, not a sentence to synthesize -
                    # just forward it in place so it stays ordered relative
                    # to any sentence queued ahead of it.
                    clip_queue.put(("confirm", item[1:]))
                    continue
                audio, sr = synthesize(client, item)
                clip_queue.put(("clip", (item, audio, sr)))

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()

        first_clip_at = None
        t0 = time.time()
        while True:
            item = clip_queue.get()
            if item is None:
                break
            kind, payload = item
            if kind == "confirm":
                if first_clip_at is None:
                    # The confirmation question is genuinely the first thing
                    # spoken this turn - count it, so the timer below doesn't
                    # make a real, already-spoken prompt look like dead air.
                    first_clip_at = time.time()
                    log(f"[first audio after {first_clip_at - t0:.1f}s]")
                    events.publish("first_audio", seconds=round(first_clip_at - t0, 1))
                description, response_box = payload
                response_box.put(
                    confirm_with_user(client, listening_enabled, audio_queue, events, vad, whisper, description)
                )
                continue
            sentence, audio, sr = payload
            if first_clip_at is None:
                first_clip_at = time.time()
                log(f"[first audio after {first_clip_at - t0:.1f}s]")
                events.publish("first_audio", seconds=round(first_clip_at - t0, 1))
            log(f"Gabbie: {sentence}")
            events.publish("speaking", text=sentence)
            sd.play(audio, sr)
            sd.wait()

        if first_clip_at is None:
            # Both attempts came back empty - don't leave the user met with
            # silence and wondering if she heard them at all.
            fallback = "Sorry, could you say that again?"
            log(f"Gabbie: {fallback}  [fallback after empty reply]")
            events.publish("speaking", text=fallback)
            audio, sr = synthesize(client, fallback)
            sd.play(audio, sr)
            sd.wait()
            full_text["reply"] = fallback

        time.sleep(PLAYBACK_COOLDOWN_SECONDS)
        events.publish("done_speaking", text=full_text.get("reply", ""))
        return full_text.get("reply", "")
    finally:
        drain_queue(audio_queue)
        listening_enabled.set()


def main():
    log("Loading speech-to-text model...")
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    client = OpenAI(base_url=NEURALFORGE_URL, api_key="not-needed")
    vad = webrtcvad.Vad(VAD_MODE)
    memory_lock = threading.Lock()

    events = EventBus()
    if events.start():
        log(f"[events broadcasting on {events.host}:{events.port}]")
    else:
        log("[event port unavailable, running without event broadcast]")

    audio_queue = queue.Queue()
    listening_enabled = threading.Event()
    listening_enabled.set()

    remembered_facts = load_memory()
    if remembered_facts:
        log(f"Loaded {len(remembered_facts)} remembered fact(s).")
    messages = [{"role": "system", "content": build_system_prompt(remembered_facts)}]
    transcript_path = open_transcript()
    awake_until = 0.0
    remember_thread = None
    last_extracted_index = len(messages)
    log("Gabbie is listening... (Ctrl+C to quit)")

    def flush_memory_async():
        nonlocal remember_thread, last_extracted_index
        if remember_thread and remember_thread.is_alive():
            return  # one already in flight; it'll pick up next time
        if len(messages) <= last_extracted_index:
            return
        context = messages[last_extracted_index:]
        last_extracted_index = len(messages)
        remember_thread = threading.Thread(
            target=remember_worker, args=(client, memory_lock, events, context), daemon=True
        )
        remember_thread.start()

    def flush_memory_sync():
        nonlocal last_extracted_index
        if remember_thread:
            remember_thread.join()
        if len(messages) > last_extracted_index:
            remember_worker(client, memory_lock, events, messages[last_extracted_index:])
            last_extracted_index = len(messages)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=make_input_callback(audio_queue, listening_enabled),
    )

    with stream:
        try:
            while True:
                log("[listening]")
                events.publish("listening")
                audio = listen_for_utterance(vad, audio_queue)
                if audio.size < SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
                    continue

                log("[transcribing]")
                events.publish("transcribing")
                t0 = time.time()
                text = transcribe(whisper, audio)
                if not text:
                    continue

                already_awake = time.time() < awake_until
                if not already_awake and not contains_wake_word(text):
                    log(f"You: {text}  ({time.time() - t0:.1f}s)  [no wake word, ignored]")
                    events.publish("heard", text=text, ignored=True)
                    continue
                log(f"You: {text}  ({time.time() - t0:.1f}s)" + ("" if already_awake else "  [woke up]"))
                events.publish("heard", text=text, woke=not already_awake)

                append_transcript(transcript_path, "user", text)

                if GOODBYE_PATTERN.search(text):
                    speak(client, listening_enabled, audio_queue, events, "Bye bye!")
                    append_transcript(transcript_path, "assistant", "Bye bye!")
                    flush_memory_sync()
                    events.publish("bye")
                    break

                if contains_sleep_phrase(text):
                    log("[going to sleep]")
                    events.publish("sleeping")
                    speak(client, listening_enabled, audio_queue, events, "Okay, I'll be quiet.")
                    append_transcript(transcript_path, "assistant", "Okay, I'll be quiet.")
                    awake_until = 0.0
                    # Nobody's actively waiting on anything right now, so
                    # catch up on memory in the background while she's quiet.
                    flush_memory_async()
                    continue

                # neuralforge's LLM only has one inference slot, so a memory
                # extraction call left running would otherwise queue up
                # behind (or in front of) this one and blow out latency.
                if remember_thread and remember_thread.is_alive():
                    log("[waiting for memory update to finish]")
                    remember_thread.join()

                messages.append({"role": "user", "content": text})
                log("[thinking + speaking]")
                events.publish("thinking")
                t0 = time.time()
                reply = think_and_speak(
                    client, listening_enabled, audio_queue, events, vad, whisper, messages
                )
                # Start the awake window now that the mic is live again,
                # rather than from when the user last spoke - otherwise a
                # long reply (mic muted the whole time) eats into their
                # reply window before they even get a chance to respond.
                awake_until = time.time() + AWAKE_WINDOW_SECONDS
                messages.append({"role": "assistant", "content": reply})
                append_transcript(transcript_path, "assistant", reply)
                log(f"[done speaking]  ({time.time() - t0:.1f}s total)")
        except KeyboardInterrupt:
            flush_memory_sync()
            raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Bye!")
