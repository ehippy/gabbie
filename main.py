import collections
import io
import queue
import re
import threading
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

from faster_whisper import WhisperModel
from openai import OpenAI
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

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
    "You are Gabbie, a warm, playful voice assistant. Keep replies short "
    "and conversational since they will be read aloud."
)
GOODBYE_WORDS = ("goodbye", "bye", "exit", "quit")
GOODBYE_PATTERN = re.compile(r"\b(" + "|".join(GOODBYE_WORDS) + r")\b", re.IGNORECASE)
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


def speak(client, listening_enabled, audio_queue, text):
    """Speak a short, fixed line (no streaming needed)."""
    listening_enabled.clear()
    drain_queue(audio_queue)
    try:
        audio, sr = synthesize(client, text)
        sd.play(audio, sr)
        sd.wait()
        time.sleep(PLAYBACK_COOLDOWN_SECONDS)
    finally:
        drain_queue(audio_queue)
        listening_enabled.set()


def think_and_speak(client, listening_enabled, audio_queue, messages):
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
            parts = []
            buffer = ""
            llm_stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                # This model "thinks" before answering by default, which
                # roughly doubles latency for zero benefit on casual chat.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            for chunk in llm_stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                parts.append(delta)
                buffer += delta
                match = SENTENCE_BOUNDARY.search(buffer)
                while match:
                    sentence = buffer[: match.end()].strip()
                    buffer = buffer[match.end() :]
                    if sentence:
                        sentence_queue.put(sentence)
                    match = SENTENCE_BOUNDARY.search(buffer)
            if buffer.strip():
                sentence_queue.put(buffer.strip())
            sentence_queue.put(None)
            full_text["reply"] = "".join(parts).strip()

        def tts_worker():
            while True:
                sentence = sentence_queue.get()
                if sentence is None:
                    clip_queue.put(None)
                    return
                clip_queue.put(synthesize(client, sentence))

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
            audio, sr = clip
            sd.play(audio, sr)
            sd.wait()

        time.sleep(PLAYBACK_COOLDOWN_SECONDS)
        return full_text.get("reply", "")
    finally:
        drain_queue(audio_queue)
        listening_enabled.set()


def main():
    log("Loading speech-to-text model...")
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    client = OpenAI(base_url=NEURALFORGE_URL, api_key="not-needed")
    vad = webrtcvad.Vad(VAD_MODE)

    audio_queue = queue.Queue()
    listening_enabled = threading.Event()
    listening_enabled.set()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    awake_until = 0.0
    log("Gabbie is listening... (Ctrl+C to quit)")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=make_input_callback(audio_queue, listening_enabled),
    )

    with stream:
        while True:
            log("[listening]")
            audio = listen_for_utterance(vad, audio_queue)
            if audio.size < SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
                continue

            log("[transcribing]")
            t0 = time.time()
            text = transcribe(whisper, audio)
            if not text:
                continue

            already_awake = time.time() < awake_until
            if not already_awake and not contains_wake_word(text):
                log(f"You: {text}  ({time.time() - t0:.1f}s)  [no wake word, ignored]")
                continue
            awake_until = time.time() + AWAKE_WINDOW_SECONDS
            log(f"You: {text}  ({time.time() - t0:.1f}s)" + ("" if already_awake else "  [woke up]"))

            if GOODBYE_PATTERN.search(text):
                speak(client, listening_enabled, audio_queue, "Bye bye!")
                break

            messages.append({"role": "user", "content": text})
            log("[thinking + speaking]")
            t0 = time.time()
            reply = think_and_speak(client, listening_enabled, audio_queue, messages)
            messages.append({"role": "assistant", "content": reply})
            log(f"Gabbie: {reply}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Bye!")
