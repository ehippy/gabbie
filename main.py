import collections
import io
import json
import queue
import re
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

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
    "have no memory at all."
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
        if m["role"] in ("user", "assistant")
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


def think_and_speak(client, listening_enabled, audio_queue, events, messages):
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
            for attempt in range(2):
                parts = []
                buffer = ""
                llm_stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
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
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    parts.append(delta)
                    buffer += delta
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
                full_reply = re.sub(r"<think>.*$", "", full_reply, flags=re.DOTALL)
                full_text["reply"] = full_reply.strip()
                if full_text["reply"]:
                    break
                log("[empty reply, retrying]")
            sentence_queue.put(None)

        def tts_worker():
            while True:
                sentence = sentence_queue.get()
                if sentence is None:
                    clip_queue.put(None)
                    return
                audio, sr = synthesize(client, sentence)
                clip_queue.put((sentence, audio, sr))

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()

        first_clip_at = None
        t0 = time.time()
        while True:
            clip = clip_queue.get()
            if clip is None:
                break
            if first_clip_at is None:
                first_clip_at = time.time()
                log(f"[first audio after {first_clip_at - t0:.1f}s]")
                events.publish("first_audio", seconds=round(first_clip_at - t0, 1))
            sentence, audio, sr = clip
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
                reply = think_and_speak(client, listening_enabled, audio_queue, events, messages)
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
