import base64
import collections
import html.parser
import io
import json
import os
import queue
import random
import re
import subprocess
import sys
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
DEFAULT_LLM_MODEL = "Qwen3.8-27B-GGUF-UD-Q4_K_XL"
TTS_MODEL = "kokoro-v1"
TTS_VOICE = "af_heart"
SYSTEM_PROMPT = (
    "You are Gabbie, a warm, playful voice assistant, speaking out loud in "
    "a live conversation. Never use markdown or any other text formatting "
    "- no asterisks or underscores for emphasis, no bullet points or "
    "numbered lists, no headers, no code fences or backticks. It's all "
    "read out loud exactly as written, symbols included, so write "
    "everything as plain spoken sentences; convey emphasis the way you "
    "would out loud, through word choice and phrasing, not formatting. "
    "Match your reply length to the moment: a simple "
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
    "the conversation. You have long-term memory tools - remember, "
    "list_memories, edit_memory, forget_memory - covered below; if you "
    "don't know something about the user yet, say so plainly rather than "
    "claiming you have no memory at all. You also have tools to check the "
    "current date/time, manage that long-term memory, list/read files on "
    "the user's computer, search the web, fetch a specific web page, "
    "search for images, actually look at one of those images, write or "
    "edit files, and run a Python script you've written with run_python. "
    "Prefer writing and running a script over doing math or date logic "
    "in your head - that's exactly the kind of thing you get wrong that "
    "a script won't; write it to the scratch folder mentioned in "
    "run_python's own description and just run it, no need to mention "
    "the file to the user at all unless they ask. Use remember in the "
    "moment whenever the user shares "
    "something worth keeping long-term (name, preferences, ongoing "
    "projects, decisions) - this doesn't happen on its own, so don't wait "
    "or assume it's already been saved. If something you knew turns out "
    "to be wrong or outdated, fix it with edit_memory or forget_memory "
    "rather than leaving stale info around; use list_memories first if "
    "you need the exact wording to do that. image_search only surfaces results "
    "on the local dashboard - you don't see them from that alone, so just "
    "acknowledge you've pulled them up rather than describing them. If the "
    "user then asks you to look at, describe, compare, or react to one of "
    "them, use view_image to actually see it instead of guessing. "
    "When you use a tool, "
    "summarize what's useful from the result in your own words rather than "
    "reciting it verbatim - especially file contents or web pages, which "
    "can be long. Writing or editing a file outside that scratch folder "
    "always asks the user to confirm out loud before it actually "
    "happens, automatically - so just call the tool directly, don't ask "
    "for confirmation yourself first, that would just double up. "
    "Nothing in the scratch folder needs confirming, including "
    "run_python, since it's a low-stakes area just for scripts like that."
)
MOOD_EXTRACTION_PROMPT = (
    "You score how a conversation felt for Gabbie, a voice assistant. "
    "You'll be given a recent conversation excerpt. Decide a mood_delta: "
    "how this conversation felt for Gabbie herself, as an integer from -2 "
    "to 2. Warmth, compliments, playfulness, or a fun exchange should "
    "nudge it positive; rudeness, dismissiveness, or curtness directed at "
    "her should nudge it negative. Use 0 for a neutral or purely "
    "transactional exchange - don't invent a shift that isn't there.\n"
    'Reply with ONLY JSON: {"mood_delta": 0}.'
)
BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
# Which transcript file is "active" - persists across a plain process
# restart (crash, --watch, Ctrl+C) so the dashboard doesn't see that as a
# new session; only start_new_transcript()/end_transcript_session() touch it.
TRANSCRIPT_POINTER_PATH = BASE_DIR / "current_transcript.txt"
# Where run_python executes from - deliberately not "anywhere in $HOME"
# like read_file/write_file. This is the one tool that runs code rather
# than just moving text around, so it gets an actual boundary: only
# scripts written here (via write_file first) can be run at all.
SCRATCH_DIR = BASE_DIR / "scratch"
RUN_PYTHON_TIMEOUT_SECONDS = 10
MEMORY_PATH = BASE_DIR / "memory.json"
# Existence alone is the signal - dashboard.py's "New Conversation" button
# touches this file; main.py's loop notices it between turns and clears it.
RESET_FLAG_PATH = BASE_DIR / "reset_requested.flag"
MOOD_PATH = BASE_DIR / "mood.json"
MOOD_MIN, MOOD_MAX = -5, 5
MOOD_HALF_LIFE_HOURS = 6  # how fast an untouched mood drifts back to neutral
SETTINGS_PATH = BASE_DIR / "settings.json"
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


def strip_markdown(text):
    """Kokoro (and every OpenAI-compatible TTS endpoint like it) has no
    concept of markdown - it just reads the literal characters, so
    "**important**" comes out as "asterisk asterisk important asterisk
    asterisk" instead of anything resembling emphasis. There's no markup
    that gets Kokoro to actually emphasize a word either, so this is
    pure cleanup, not a substitute for real emphasis - the system prompt
    is the primary defense (asking the model not to use markdown at
    all); this is the fallback for whatever slips through anyway."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


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
    # Whisper is well known to hallucinate text on silence/near-silence,
    # and initial_prompt makes it worse here specifically - it conditions
    # the decoder on "Gabbie" as prior context, so a quiet room that
    # barely trips the VAD upstream can come back as "Gabbie Gabbie" with
    # nobody having said anything. hotwords biases recognition toward the
    # wake word the same way without that conditioning effect, vad_filter
    # runs a second, more precise VAD pass to skip non-speech chunks
    # before decoding at all, and the no_speech_prob check below is a last
    # line of defense against whatever gets through anyway.
    segments, _ = whisper.transcribe(
        audio, language="en", hotwords="Gabbie", vad_filter=True
    )
    return " ".join(
        segment.text.strip() for segment in segments if segment.no_speech_prob < 0.6
    ).strip()


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


def load_mood():
    """Current mood, decayed toward neutral (0) based on how long it's
    been since the last update - so a good or bad mood fades on its own
    rather than sticking around forever."""
    if not MOOD_PATH.exists():
        return 0.0
    try:
        data = json.loads(MOOD_PATH.read_text())
        value = float(data["value"])
        ts = datetime.fromisoformat(data["ts"])
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return 0.0
    elapsed_hours = max(0.0, (datetime.now() - ts).total_seconds() / 3600)
    return value * (0.5 ** (elapsed_hours / MOOD_HALF_LIFE_HOURS))


def save_mood(value):
    value = max(MOOD_MIN, min(MOOD_MAX, value))
    MOOD_PATH.write_text(json.dumps({"value": round(value, 2), "ts": datetime.now().isoformat()}))
    return value


def mood_label(value):
    """None means near-neutral - not worth mentioning in the prompt or
    showing as anything but a plain, wordless indicator on the dashboard."""
    if value >= 3:
        return "cheerful and extra playful"
    if value >= 1:
        return "in a good mood"
    if value <= -3:
        return "a bit low, but still warm and helpful"
    if value <= -1:
        return "a little subdued"
    return None


def build_system_prompt(facts, mood=0.0):
    prompt = SYSTEM_PROMPT
    label = mood_label(mood)
    if label:
        prompt += (
            f"\n\nRight now you're feeling {label} - let that color your "
            "tone naturally without announcing it, unless the user asks "
            "how you're doing."
        )
    if facts:
        bullets = "\n".join(f"- {f['fact']}" for f in facts)
        prompt += f"\n\nThings you remember from previous conversations:\n{bullets}"
    return prompt


def load_settings():
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(**updates):
    settings = load_settings()
    settings.update(updates)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return settings


def current_llm_model():
    # Read fresh rather than cached, so a change from the dashboard takes
    # effect on the next LLM call instead of needing a restart.
    return load_settings().get("llm_model") or DEFAULT_LLM_MODEL


def open_transcript():
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}.jsonl"
    return path


def start_new_transcript():
    """A real new session: an explicit reset, or right after saying
    goodbye. Rotates to a fresh transcript file and points
    TRANSCRIPT_POINTER_PATH at it."""
    path = open_transcript()
    TRANSCRIPT_POINTER_PATH.write_text(path.name)
    return path


def resume_or_start_transcript():
    """Reuse the transcript from before this process started (a plain
    restart - a crash, --watch picking up a code change, Ctrl+C) instead
    of rotating to a new file, so the dashboard's hydration (which only
    ever looks at the most recent transcript) doesn't make everything
    from before the restart look like it vanished. Only an explicit
    reset or "goodbye" actually end a conversation and clear the
    pointer - a mere process restart isn't a new session."""
    if TRANSCRIPT_POINTER_PATH.exists():
        name = TRANSCRIPT_POINTER_PATH.read_text().strip()
        if name:
            # Trust the name without checking the file exists yet -
            # append_transcript() creates it lazily on first write, so a
            # reset immediately followed by a restart (nothing said yet)
            # would otherwise look like an invalid pointer here.
            return TRANSCRIPTS_DIR / name
    return start_new_transcript()


def end_transcript_session():
    TRANSCRIPT_POINTER_PATH.unlink(missing_ok=True)


def append_transcript(path, role, content):
    entry = {"ts": datetime.now().isoformat(), "role": role, "content": content}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def extract_mood_delta(client, recent_messages):
    # A single isolated exchange often doesn't make sense on its own, so
    # this gets a few turns of context instead of just the latest.
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gabbie'}: {m['content']}"
        for m in recent_messages
        # A tool-calling turn's assistant message often has content=None
        # (the model went straight for the tool with nothing to say first)
        # - skip those rather than feeding the extractor literal "Gabbie:
        # None" lines.
        if m["role"] in ("user", "assistant") and m.get("content")
    )
    try:
        response = client.chat.completions.create(
            model=current_llm_model(),
            messages=[
                {"role": "system", "content": MOOD_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Recent conversation:\n{convo}"},
            ],
            temperature=0.1,
            # Unlike the conversational reply path, this runs off the
            # latency-critical path (see the join() in main()), so it's
            # worth leaving "thinking" enabled here.
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return max(-2, min(2, int(data.get("mood_delta", 0))))
    except Exception:
        return 0


def mood_worker(client, events, recent_messages):
    mood_delta = extract_mood_delta(client, recent_messages)
    if not mood_delta:
        return
    new_mood = save_mood(load_mood() + mood_delta)
    log(f"[mood] {mood_delta:+d} -> {new_mood:.1f}")
    events.publish("mood", value=round(new_mood, 2), label=mood_label(new_mood))


MAX_TOOL_ROUNDS = 3
MAX_FILE_CHARS = 4000
MAX_DIRECTORY_ENTRIES = 100
MAX_FETCH_CHARS = 6000
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"
BRAVE_MAX_RETRIES = 3


def _brave_get(url, params):
    """GET against one of Brave's search endpoints, retrying on a 429 -
    the Free plan allows only 1 request/second, so two tool calls back to
    back (e.g. a web search right after an image search) can trip it
    easily. Returns (response, error_message); error_message is only set
    on a network-level failure, not an HTTP error status - callers still
    check response.status_code themselves."""
    response = None
    for attempt in range(BRAVE_MAX_RETRIES):
        try:
            response = httpx.get(
                url,
                params=params,
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                timeout=10,
            )
        except httpx.HTTPError as e:
            return None, f"Error contacting search: {e}"
        if response.status_code != 429:
            return response, None
        if attempt < BRAVE_MAX_RETRIES - 1:
            # x-ratelimit-reset lists seconds-until-reset per policy (the
            # per-second limit first) - fall back to a flat 1.1s if it's
            # missing rather than guessing something longer.
            reset = response.headers.get("x-ratelimit-reset", "")
            try:
                wait = float(reset.split(",")[0]) + 0.1
            except ValueError:
                wait = 1.1
            log(f"[brave rate-limited, retrying in {wait:.1f}s]")
            time.sleep(wait)
    return response, None
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
            "name": "remember",
            "description": "Save a new fact to long-term memory. Do this "
            "in the moment when the user shares something worth "
            "remembering long-term (name, preferences, ongoing projects, "
            "decisions) - don't wait for the conversation to end, and "
            "don't rely on it happening on its own. Make the fact "
            "self-contained (resolve pronouns and context) since it'll be "
            "read back later with no surrounding conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, as a standalone statement"}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List everything currently remembered about "
            "the user - use this to check what's already saved, avoid "
            "saving a duplicate, or find the exact wording of something "
            "before editing or forgetting it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_memory",
            "description": "Correct an existing memory - use when a "
            "previously remembered fact turns out to be outdated or "
            "wrong. old_fact must match a remembered fact exactly (use "
            "list_memories first to get the exact wording).",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_fact": {"type": "string", "description": "The exact existing fact to replace"},
                    "new_fact": {"type": "string", "description": "The corrected fact"},
                },
                "required": ["old_fact", "new_fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Permanently remove a remembered fact - use "
            "when the user asks you to forget something, or a fact is no "
            "longer true and doesn't need a replacement. fact must match "
            "a remembered fact exactly (use list_memories first to get "
            "the exact wording).",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The exact fact to forget"}},
                "required": ["fact"],
            },
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
            "results are visual only, not something to describe back unless "
            "the user then asks you to look at one with view_image. Returns "
            "every decently-sized result there is for the query, not just a "
            "sample - there's no pagination or 'next page' to fetch after "
            "this, so don't offer one. If nothing decent turns up, a "
            "narrower or different query is the only way to get more.",
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
            "name": "view_image",
            "description": "Actually look at one specific image from the "
            "most recent image_search results and describe or react to "
            "what's in it. Use this only when the user asks you to look at, "
            "describe, compare, or react to a specific picture (e.g. 'what "
            "do you think of the second one', 'describe that first image') "
            "- don't call it just because image_search found results, and "
            "it only works after image_search has already run this "
            "conversation. It takes noticeably longer than image_search "
            "since it has to actually process the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Which image, counting from 1 for the first result, 2 for the second, etc.",
                    }
                },
                "required": ["index"],
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
            f"content on the user's computer. Asks the user to confirm out "
            f"loud first, except inside {SCRATCH_DIR} (see run_python) - "
            "that folder is a low-stakes scratch area.",
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
            "existing file on the user's computer with new text. Asks the "
            f"user to confirm out loud first, except inside {SCRATCH_DIR} "
            "(see run_python) - that folder is a low-stakes scratch area.",
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
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python script and return its printed "
            "output, optionally passing it command-line arguments. Use "
            "this to actually execute logic you've written (like a date "
            "calculation) instead of working it out yourself - mental "
            "math and date arithmetic are exactly the kind of thing you "
            "get wrong that a script won't. If a script takes an input "
            "that'll vary (a date, a name), give its __main__ block a "
            "sys.argv parameter ONCE when you write it, then reuse that "
            "same script via args for every future input - never write a "
            "new one-off file just to supply a different value to logic "
            f"you've already written. Only scripts under {SCRATCH_DIR} "
            "can be run - write the script there with write_file first. "
            "Neither call needs the user to confirm out loud - that "
            "folder is a low-stakes scratch area, unlike writing or "
            "editing a file anywhere else.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": f"Path to the .py file, under {SCRATCH_DIR}"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Command-line arguments to pass to the script, e.g. ["2026-12-08"]',
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# Tools that touch a file outside the low-stakes scratch sandbox get a
# spoken confirmation first, since voice input is lossy and there's no
# click-to-confirm UI to catch a mis-transcribed request before it does
# something irreversible. Scratch itself (run_python's whole world, plus
# write_file/edit_file when the model happens to target something in
# there) skips it - see needs_confirmation() - since a script that can
# only ever live and run inside that one confined directory isn't the
# kind of mistake voice confirmation is protecting against, and gating
# "write a helper script, then run it" behind two separate confirmations
# for one logical action was just friction with no real safety payoff.
CONFIRMABLE_TOOLS = {"write_file", "edit_file"}


def _resolve_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path.home() / p


def needs_confirmation(name, kwargs):
    if name not in CONFIRMABLE_TOOLS:
        return False
    path = kwargs.get("path")
    if not path:
        return True  # malformed call - err toward confirming
    try:
        _resolve_path(path).resolve().relative_to(SCRATCH_DIR.resolve())
        return False
    except ValueError:
        return True


def tool_get_current_datetime(**_kwargs):
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


def tool_remember(fact=None, events=None, **_kwargs):
    if not fact or not fact.strip():
        return "Error: no fact given"
    fact = fact.strip()
    existing = load_memory()
    if any(f["fact"] == fact for f in existing):
        return f"Already remembered: {fact}"
    existing.append({"fact": fact, "ts": datetime.now().isoformat()})
    MEMORY_PATH.write_text(json.dumps(existing, indent=2))
    log(f"[remembered] {fact}")
    if events:
        events.publish("remembered", fact=fact)
    return f"Remembered: {fact}"


def tool_list_memories(**_kwargs):
    facts = load_memory()
    if not facts:
        return "No memories saved yet."
    return "\n".join(f"- {f['fact']}" for f in facts)


def tool_edit_memory(old_fact=None, new_fact=None, events=None, **_kwargs):
    if not old_fact or not new_fact:
        return "Error: need both old_fact and new_fact"
    old_fact, new_fact = old_fact.strip(), new_fact.strip()
    facts = load_memory()
    matches = [f for f in facts if f["fact"] == old_fact]
    if not matches:
        return f"Error: no memory matching '{old_fact}' - use list_memories to get the exact wording."
    if len(matches) > 1:
        return f"Error: '{old_fact}' matches more than one memory - this shouldn't normally happen."
    for f in facts:
        if f["fact"] == old_fact:
            f["fact"] = new_fact
            f["ts"] = datetime.now().isoformat()
            break
    MEMORY_PATH.write_text(json.dumps(facts, indent=2))
    log(f"[remembered] {old_fact} -> {new_fact}")
    if events:
        events.publish("remembered", fact=new_fact)
    return f"Updated memory: '{old_fact}' is now '{new_fact}'"


def tool_forget_memory(fact=None, events=None, **_kwargs):
    if not fact:
        return "Error: no fact given"
    fact = fact.strip()
    facts = load_memory()
    matches = [f for f in facts if f["fact"] == fact]
    if not matches:
        return f"Error: no memory matching '{fact}' - use list_memories to get the exact wording."
    if len(matches) > 1:
        return f"Error: '{fact}' matches more than one memory - this shouldn't normally happen."
    remaining = [f for f in facts if f["fact"] != fact]
    MEMORY_PATH.write_text(json.dumps(remaining, indent=2))
    log(f"[forgot] {fact}")
    if events:
        events.publish("forgot", fact=fact)
    return f"Forgot: {fact}"


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
    response, error = _brave_get(BRAVE_SEARCH_URL, {"q": query, "count": 5, "safesearch": "off"})
    if error:
        return error
    if response.status_code == 429:
        return "Error: search is rate-limited right now - wait a few seconds and try again."
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


MIN_IMAGE_DIMENSION = 400  # px, on the shorter side - filters out icons/logos/tiny thumbnails
# Brave's Image Search has no pagination at all - no offset param, and
# their own docs say to just raise count instead. 100 is their
# documented max, so this is the largest single batch obtainable; there
# is no "next page" to fetch beyond it.
BRAVE_IMAGE_SEARCH_COUNT = 100
# Results from the most recent image_search, so a later "look at the second
# one" doesn't need the model to pass a URL around - single conversation,
# single process, so plain module state is enough.
_last_image_results = []


def tool_image_search(query=None, **_kwargs):
    if not query:
        return "Error: no query given"
    if not BRAVE_API_KEY:
        return "Error: image search isn't set up yet - no BRAVE_API_KEY configured."
    # Brave doesn't offer a server-side size filter, so over-fetch (its max
    # - see BRAVE_IMAGE_SEARCH_COUNT) and filter by actual source
    # dimensions below - otherwise a query dominated by small icons/logos
    # could come back mostly filtered out.
    # safesearch=off because Brave's default (moderate) returns zero
    # results outright for some queries rather than just filtering them -
    # this is a personal local assistant with no separate content policy
    # to defer to.
    response, error = _brave_get(
        BRAVE_IMAGE_SEARCH_URL, {"q": query, "count": BRAVE_IMAGE_SEARCH_COUNT, "safesearch": "off"}
    )
    if error:
        return error
    if response.status_code == 429:
        return "Error: image search is rate-limited right now - wait a few seconds and try again."
    if response.status_code != 200:
        return f"Error: image search returned HTTP {response.status_code}"
    results = response.json().get("results", [])
    structured = []
    for r in results:
        thumbnail = r.get("thumbnail", {}).get("src")
        if not thumbnail:
            continue  # nothing to show for this one, skip it
        properties = r.get("properties", {})
        width, height = properties.get("width", 0), properties.get("height", 0)
        if min(width, height) < MIN_IMAGE_DIMENSION:
            continue  # too small to be a "decent" picture - likely an icon/logo/thumbnail
        structured.append(
            {
                "title": r.get("title", ""),
                "source_url": r.get("url", ""),
                "thumbnail": thumbnail,
                # Full-resolution image for the dashboard's lightbox - hosted
                # on the original site rather than Brave's thumbnail proxy,
                # so it's more likely to be hotlink-protected; the lightbox
                # falls back to the thumbnail if it fails to load.
                "image_url": properties.get("url", ""),
            }
        )
    if not structured:
        return f"No decently-sized image results for '{query}' ({len(results)} results total, none met the size bar)."
    # The model can't actually see these - it's told just enough to
    # acknowledge the request without inventing a description of images it
    # never received. Also tells it the raw-vs-kept counts so it can
    # honestly convey "there were more, most just weren't decent-sized"
    # instead of implying a fetchable next page - Brave's image search has
    # no pagination at all, this single batch is the whole result set.
    text = (
        f"Found {len(structured)} decently-sized images for '{query}' (out of {len(results)} "
        "total results) - shown on the dashboard, not spoken. This is the whole result set - "
        "there's no further page to fetch for this query."
    )
    global _last_image_results
    _last_image_results = structured
    return text, {"query": query, "results": structured}


VIEW_IMAGE_PROMPT = (
    "Describe what's shown in this image factually and specifically - "
    "subject, setting, notable details - in 2-3 sentences. Someone else "
    "will paraphrase your description out loud, so be accurate and "
    "concrete rather than trying to sound conversational yourself."
)


def tool_view_image(index=None, client=None, **_kwargs):
    if not _last_image_results:
        return "Error: no recent image search results to look at - run image_search first."
    try:
        position = int(index) - 1
    except (TypeError, ValueError):
        return "Error: index must be a number - 1 for the first image, 2 for the second, etc."
    if not 0 <= position < len(_last_image_results):
        return f"Error: only {len(_last_image_results)} image(s) available from the last search."
    image = _last_image_results[position]
    for url in (image.get("image_url"), image.get("thumbnail")):
        if not url:
            continue
        try:
            response = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
            response.raise_for_status()
            break
        except httpx.HTTPError:
            continue  # full-res source may block hotlinking - fall back to the thumbnail
    else:
        return "Error: couldn't load that image to look at it."
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    data_url = f"data:{content_type};base64,{base64.b64encode(response.content).decode()}"
    try:
        vision_response = client.chat.completions.create(
            model=current_llm_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VIEW_IMAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.6,
            # Same fix as the main reply loop: this model "thinks" by
            # default, which for a fixed max_tokens budget can burn the
            # whole thing on hidden reasoning and leave zero visible
            # content (finish_reason "length", empty text) - seen directly
            # while testing this against a real vision call.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        description = (vision_response.choices[0].message.content or "").strip()
        if not description:
            return "Error: the model didn't return a description for that image - try again."
        return description, {
            "index": position + 1,
            "title": image.get("title", ""),
            "thumbnail": image.get("thumbnail", ""),
        }
    except Exception as e:
        return f"Error: the current model couldn't process the image ({e}) - try a vision-capable model in Settings."


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


def tool_run_python(path=None, args=None, **_kwargs):
    if not path:
        return "Error: no path given"
    SCRATCH_DIR.mkdir(exist_ok=True)
    target = _resolve_path(path).resolve()
    try:
        target.relative_to(SCRATCH_DIR.resolve())
    except ValueError:
        return f"Error: run_python can only run scripts under {SCRATCH_DIR} - write it there with write_file first."
    if not target.exists():
        return f"Error: no such file: {target}"
    if target.suffix != ".py":
        return "Error: run_python only runs .py files"
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return "Error: args must be a list of strings"
    try:
        result = subprocess.run(
            [sys.executable, str(target), *args],
            capture_output=True,
            text=True,
            timeout=RUN_PYTHON_TIMEOUT_SECONDS,
            cwd=SCRATCH_DIR,
        )
    except subprocess.TimeoutExpired:
        return f"Error: script timed out after {RUN_PYTHON_TIMEOUT_SECONDS}s - it may be stuck in a loop."
    except OSError as e:
        return f"Error running script: {e}"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    if len(output) > MAX_FILE_CHARS:
        output = output[:MAX_FILE_CHARS] + f"\n\n[...truncated, {len(output)} characters total]"
    status = "" if result.returncode == 0 else f" (exited with code {result.returncode})"
    return f"Output{status}:\n{output}"


def describe_tool_action(name, kwargs):
    path = kwargs.get("path", "(unknown path)")
    if name == "write_file":
        return f"I want to write to the file {path}."
    if name == "edit_file":
        return f"I want to edit the file {path}."
    if name == "run_python":
        return f"I want to run the script {path}."
    return f"I want to run {name}."


TOOL_FUNCTIONS = {
    "get_current_datetime": tool_get_current_datetime,
    "remember": tool_remember,
    "list_memories": tool_list_memories,
    "edit_memory": tool_edit_memory,
    "forget_memory": tool_forget_memory,
    "list_directory": tool_list_directory,
    "read_file": tool_read_file,
    "web_search": tool_web_search,
    "image_search": tool_image_search,
    "view_image": tool_view_image,
    "fetch_url": tool_fetch_url,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "run_python": tool_run_python,
}


def execute_tool(name, arguments_json, client=None, events=None):
    """Returns (text_for_the_model, extra_for_display). extra_for_display is
    None unless the tool has something worth showing visually (e.g. search
    results) - most tools just return a plain string, normalized here.
    client is only used by view_image (needs it for the vision call) and
    events only by the memory tools (to notify the dashboard) - every
    other tool ignores both via its **_kwargs catch-all."""
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        kwargs = {}
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'", None
    try:
        result = func(client=client, events=events, **kwargs)
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


def think_and_speak(client, listening_enabled, audio_queue, events, vad, whisper, messages, transcript_path):
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
                    model=current_llm_model(),
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
                        sentence = strip_markdown(buffer[: match.end()].strip())
                        buffer = buffer[match.end() :]
                        if sentence:
                            sentence_queue.put(sentence)
                        match = SENTENCE_BOUNDARY.search(buffer)
                # Guard against the stream ending mid-<think> (never closed):
                # don't speak or store the raw reasoning fragment.
                if "<think>" not in buffer and buffer.strip():
                    sentence_queue.put(strip_markdown(buffer.strip()))
                # Strip here too so a leaked <think> block (or markdown)
                # never ends up in conversation history or the transcript,
                # even though it's already kept out of what actually gets
                # spoken above.
                full_reply = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.DOTALL)
                full_reply = re.sub(r"<think>.*$", "", full_reply, flags=re.DOTALL).strip()
                full_reply = strip_markdown(full_reply)

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
                        try:
                            call_kwargs = json.loads(c["arguments"]) if c["arguments"] else {}
                        except json.JSONDecodeError:
                            call_kwargs = {}
                        if needs_confirmation(c["name"], call_kwargs):
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
                                result_text, result_extra = execute_tool(
                                    c["name"], c["arguments"], client=client, events=events
                                )
                            else:
                                result_text, result_extra = (
                                    "The user did not confirm this action, so it was not performed.",
                                    None,
                                )
                        else:
                            result_text, result_extra = execute_tool(
                                c["name"], c["arguments"], client=client, events=events
                            )
                        events.publish(
                            "tool_call",
                            name=c["name"],
                            arguments=c["arguments"],
                            result=result_text,
                            extra=result_extra,
                        )
                        # So the dashboard can rehydrate Web Activity on
                        # refresh instead of only ever showing tool calls
                        # made after the page happened to be open - same
                        # motivation as logging user/assistant turns below.
                        append_transcript(
                            transcript_path,
                            "tool_call",
                            {
                                "name": c["name"],
                                "arguments": c["arguments"],
                                "result": result_text,
                                "extra": result_extra,
                            },
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
    mood = load_mood()
    if mood_label(mood):
        log(f"[mood] starting at {mood:.1f} ({mood_label(mood)})")
    messages = [{"role": "system", "content": build_system_prompt(remembered_facts, mood)}]
    transcript_path = resume_or_start_transcript()
    awake_until = 0.0
    events.publish("awake", until=awake_until)
    mood_thread = None
    last_extracted_index = len(messages)
    # Discard any reset request left over from before this process started -
    # "New Conversation" should only ever apply to a conversation that's
    # actually in progress right now.
    RESET_FLAG_PATH.unlink(missing_ok=True)
    log("Gabbie is listening... (Ctrl+C to quit)")

    def refresh_system_prompt():
        # Memory facts change immediately when a memory tool runs, so this
        # is really only for mood, which mood_worker updates in the
        # background - rebuild the live system prompt from current disk
        # state rather than letting mood go stale until the next restart.
        messages[0]["content"] = build_system_prompt(load_memory(), load_mood())

    def flush_mood_async():
        nonlocal mood_thread, last_extracted_index
        if mood_thread and mood_thread.is_alive():
            return  # one already in flight; it'll pick up next time
        if len(messages) <= last_extracted_index:
            return
        context = messages[last_extracted_index:]
        last_extracted_index = len(messages)

        def run():
            mood_worker(client, events, context)
            refresh_system_prompt()

        mood_thread = threading.Thread(target=run, daemon=True)
        mood_thread.start()

    def flush_mood_sync():
        nonlocal last_extracted_index
        if mood_thread:
            mood_thread.join()
        if len(messages) > last_extracted_index:
            mood_worker(client, events, messages[last_extracted_index:])
            last_extracted_index = len(messages)
            refresh_system_prompt()

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
                if RESET_FLAG_PATH.exists():
                    RESET_FLAG_PATH.unlink()
                    log("[new conversation requested]")
                    flush_mood_sync()  # don't lose the mood shift from the conversation being ended
                    messages[:] = [{"role": "system", "content": build_system_prompt(load_memory(), load_mood())}]
                    last_extracted_index = len(messages)
                    transcript_path = start_new_transcript()
                    awake_until = 0.0
                    events.publish("awake", until=awake_until)
                    events.publish("new_conversation")
                    speak(client, listening_enabled, audio_queue, events, "Okay, starting fresh!")
                    append_transcript(transcript_path, "assistant", "Okay, starting fresh!")

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
                    flush_mood_sync()
                    end_transcript_session()  # a real goodbye, not just a restart - next launch starts fresh
                    events.publish("bye")
                    break

                if contains_sleep_phrase(text):
                    log("[going to sleep]")
                    events.publish("sleeping")
                    speak(client, listening_enabled, audio_queue, events, "Okay, I'll be quiet.")
                    append_transcript(transcript_path, "assistant", "Okay, I'll be quiet.")
                    awake_until = 0.0
                    events.publish("awake", until=awake_until)
                    # Nobody's actively waiting on anything right now, so
                    # catch up on mood scoring in the background while she's quiet.
                    flush_mood_async()
                    continue

                # neuralforge's LLM only has one inference slot, so a mood
                # scoring call left running would otherwise queue up behind
                # (or in front of) this one and blow out latency.
                if mood_thread and mood_thread.is_alive():
                    log("[waiting for mood update to finish]")
                    mood_thread.join()

                messages.append({"role": "user", "content": text})
                log("[thinking + speaking]")
                events.publish("thinking")
                t0 = time.time()
                reply = think_and_speak(
                    client, listening_enabled, audio_queue, events, vad, whisper, messages, transcript_path
                )
                # Start the awake window now that the mic is live again,
                # rather than from when the user last spoke - otherwise a
                # long reply (mic muted the whole time) eats into their
                # reply window before they even get a chance to respond.
                awake_until = time.time() + AWAKE_WINDOW_SECONDS
                events.publish("awake", until=awake_until)
                messages.append({"role": "assistant", "content": reply})
                append_transcript(transcript_path, "assistant", reply)
                log(f"[done speaking]  ({time.time() - t0:.1f}s total)")
        except KeyboardInterrupt:
            flush_mood_sync()
            raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Bye!")
